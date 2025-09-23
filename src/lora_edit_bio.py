#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import pdb

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
try:
    from config.paths import DATA_DIR, OUTPUTS_DIR  # noqa: F401
except Exception:
    DATA_DIR = Path("./data")
    OUTPUTS_DIR = Path("./outputs")

SEED = 10
MAX_LEN = 256
POSSIBLE_ANSWERS = ["true", "false", "n/a"]

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes"}:
        return True
    if v.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential LoRA fine-tuning on logic data, loading BioLinear ckpt into vanilla Qwen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # training behaviour
    parser.add_argument("--fine_tune", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for LoRA params")
    parser.add_argument("--batch_k", type=int, default=1, help="# same-logic training examples per step")
    parser.add_argument("--steps_per_batch", type=int, default=10, help="Gradient steps per mini-batch")

    # dataset
    parser.add_argument("--src_json", type=Path, default=DATA_DIR / "logic/level_1.json")

    # sampling (evaluation)
    parser.add_argument("--gen_k", type=int, default=1, help="# SAME-logic eval prompts per step")
    parser.add_argument("--loc_k", type=int, default=1, help="# locality eval prompts (base-correct, other logics)")

    # model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base HF model name (MUST be a model id, not a ckpt dir)")

    # ckpt from Bio pretrain
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to BioLinear-pretrained checkpoint dir (save_pretrained) or .pt")
    parser.add_argument("--strict_resume", type=str2bool, default=False, help="Strict load_state_dict (usually False)")

    # caching base correctness for locality sampling
    parser.add_argument("--cache_dir", type=Path, default=DATA_DIR / "correct", help="Cache dir for base-correct")
    parser.add_argument("--cache_base", type=str2bool, default=True, help="Use/write base-correct cache")
    parser.add_argument("--force_recompute_base", type=str2bool, default=False, help="Ignore cache and recompute")
    parser.add_argument("--base_max_new", type=int, default=6, help="max_new_tokens for base-correct inference")

    # output
    parser.add_argument("--out_root", type=Path, default=OUTPUTS_DIR / "perlogic", help="Base dir to save results")

    # misc
    parser.add_argument("--seed", type=int, default=10, help="Random seed")

    return parser.parse_args()

def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

# ---------------------------------------------------------------------------
# Read HF dir state_dict WITHOUT instantiating a model (avoid unused warnings)
# ---------------------------------------------------------------------------
def load_state_dict_from_hf_dir(dir_path: str | Path) -> dict:
    """
    Read weights from a save_pretrained directory into a single state_dict,
    WITHOUT calling from_pretrained on that directory.
    Supports:
      - model.safetensors
      - sharded: model-00001-of-0000N.safetensors
      - pytorch_model.bin / sharded *.bin
    """
    dir_path = Path(dir_path)
    sd: dict = {}

    # Try safetensors first
    try:
        from safetensors.torch import safe_open, load_file  # type: ignore
        shards = sorted(list(dir_path.glob("*.safetensors")))
        if len(shards) == 1 and shards[0].name == "model.safetensors":
            return load_file(shards[0])
        elif len(shards) >= 1:
            for shard in shards:
                with safe_open(shard, framework="pt") as f:
                    for k in f.keys():
                        sd[k] = f.get_tensor(k)
            return sd
    except Exception:
        pass  # fall back to .bin

    # pytorch .bin (possibly sharded)
    bins = sorted(list(dir_path.glob("*.bin")))
    if len(bins) == 0:
        raise FileNotFoundError(f"No weight files in {dir_path}")
    for b in bins:
        part = torch.load(b, map_location="cpu")
        if any(isinstance(v, torch.Tensor) for v in part.values()):
            # dict[name->tensor]
            for k, v in part.items():
                sd[k] = v
        else:
            # e.g. {"state_dict": {...}}
            inner = part.get("state_dict", {})
            for k, v in inner.items():
                sd[k] = v
    return sd

# ---------------------------------------------------------------------------
# BioLinear -> vanilla Linear key conversion
# ---------------------------------------------------------------------------
def convert_bio_to_vanilla_state_dict(sd: dict) -> dict:
    """
    Map:
      *.linear.weight -> *.weight
      *.linear.bias   -> *.bias
    Drop BioLinear buffers: in_coordinates/out_coordinates/*cache*.
    """
    out = {}
    drop_substrings = ("in_coordinates", "out_coordinates", "input_cache", "output_cache")
    for k, v in sd.items():
        if any(s in k for s in drop_substrings):
            continue
        if k.endswith(".linear.weight"):
            out[k[:-len(".linear.weight")] + ".weight"] = v
        elif k.endswith(".linear.bias"):
            out[k[:-len(".linear.bias")] + ".bias"] = v
        else:
            out[k] = v
    return out

def _infer_ckpt_dtype(sd: dict):
    for v in sd.values():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            return v.dtype
    return None

def _cast_state_dict(sd: dict, dtype: torch.dtype) -> dict:
    return {k: (v.to(dtype) if torch.is_tensor(v) and torch.is_floating_point(v) else v) for k, v in sd.items()}

# ---------------------------------------------------------------------------
# Encoding / generation helpers
# ---------------------------------------------------------------------------
def encode_example(row: dict, tokenizer, src_path: Path):
    # Build prompt/answer pair; mask prompt tokens in labels
    prompt = f"{row['text']} (Answer only in True, False, or N/A (Neither)). Answer:"
    answer = f"{row['label']}"
    full = prompt + " " + answer + tokenizer.eos_token

    ids = tokenizer(full, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    input_ids = ids["input_ids"].squeeze(0)
    attn_mask = ids["attention_mask"].squeeze(0)
    labels = input_ids.clone()

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0).numel()
    labels[:prompt_len] = -100
    labels[attn_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

def generate_answer(prompt: str, model, tokenizer, src_path: Path, max_new: int = 5) -> str:
    """Greedy decode then normalise to one of {true,false,n/a} (lenient)."""
    model.eval()
    with torch.no_grad():
        templ = f"{prompt} (Answer only in True, False, or N/A (Neither)). Answer:"
        ids = tokenizer(templ, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

    try:
        answer = text.split(" Answer:")[-1].strip().split()[0].lower().strip("., ")
    except Exception:
        answer = "n/a"

    if not any(k in answer for k in POSSIBLE_ANSWERS):
        lowered = text.split(" Answer:")[-1].lower()
        if any(x in lowered for x in {"yes", "true"}):
            answer = "true"
        elif any(x in lowered for x in {"not", "no", "false"}):
            answer = "false"
        elif "n/a" in lowered:
            answer = "n/a"
        else:
            answer = "v"  # default fallback
    # pdb.set_trace()
    return answer

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
CUE = re.compile(
    r"\b(?:then|which implies|this (?:would )?implies?|would suggest that|"
    r"implies?|suggests? that)\b", re.I,
)

def harvest_examples(path: Path, seed: int) -> List[dict]:
    """Flatten dataset into list of dicts: {logic, train{text,label}, eval{prompt,gold}}."""
    exs: list[dict] = []
    for rec in json.load(path.open()):
        logic = rec["question"][0]["<nl>"].strip()
        gold = str(rec["answer"]).lower()
        for dom, topics in rec.items():
            if dom in {"question", "answer"}:
                continue
            for payload in topics.values():
                nl_full = payload["<nl>"].strip()
                premise = nl_full.split("\nGiven")[0].strip()
                exs.append({
                    "logic": logic,
                    "train": {"text": premise, "label": gold},
                    "eval": {"prompt": nl_full, "gold": gold},
                })
    rng = random.Random(seed)
    rng.shuffle(exs)
    return exs

# ---------------------------------------------------------------------------
# Model & LoRA init (vanilla backbone + converted Bio ckpt)
# ---------------------------------------------------------------------------
def get_model(model_name: str, tokenizer, device: torch.device, args):
    # Safety: prevent misusing model_name as a directory (which would re-trigger warnings)
    if Path(model_name).exists() and Path(model_name).is_dir():
        raise ValueError(
            f"--model_name cannot be a directory: {model_name}\n"
            "Use a base model id (e.g., Qwen/Qwen2.5-3B-Instruct) and put your Bio ckpt path in --resume."
        )

    target_dtype = torch.bfloat16  # match your Bio pretrain dtype if needed

    print(f"[init] creating base from_pretrained(model_name={model_name})")
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=target_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # Load BioLinear ckpt (converted) into vanilla base
    if args.resume is not None:
        rp = Path(args.resume)
        print(f"[init] loading Bio ckpt (state_dict only) from {rp}")
        if rp.is_dir():
            raw_sd = load_state_dict_from_hf_dir(rp)  # NO from_pretrained on resume dir
        else:
            raw = torch.load(rp, map_location="cpu")
            raw_sd = raw.get("model_state", raw)

        sd = convert_bio_to_vanilla_state_dict(raw_sd)
        ckpt_dtype = _infer_ckpt_dtype(sd)
        if ckpt_dtype is not None and ckpt_dtype != target_dtype:
            sd = _cast_state_dict(sd, target_dtype)

        missing, unexpected = base.load_state_dict(sd, strict=args.strict_resume)
        print(f"[converted-load] missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) or len(unexpected):
            # Usually safe to proceed; print a few sample keys for debugging if needed
            pass

    # Apply LoRA to typical Qwen2 target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules
    )
    model = get_peft_model(base, lora_cfg).to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model

# ---------------------------------------------------------------------------
# Base-correct cache (for locality sampling)
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

def _base_cache_path(args: argparse.Namespace, src_file: Path) -> Path:
    dataset_tag = src_file.resolve().stem
    model_slug = _slug(args.model_name)
    ckpt_slug = _slug(args.resume.stem) if args.resume else "origin"
    key = f"{model_slug}_{ckpt_slug}_bio"
    return Path(args.cache_dir) / dataset_tag / f"base_correct_{key}.json"

def load_or_build_base_correct(
    args: argparse.Namespace,
    model,
    tokenizer,
    all_eval: List[dict],
) -> Dict[str, bool]:
    cache_file = _base_cache_path(args, args.src_json)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if args.cache_base and (not args.force_recompute_base) and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if isinstance(data, dict) and len(data) > 0:
                print(f"[cache] Loaded base-correct ({len(data)}) from {cache_file}")
                return {str(k): bool(v) for k, v in data.items()}
        except Exception as e:
            print(f"[cache] Failed to read cache ({e}), will recompute...")

    print("[cache] Computing base-correct set…")
    base_correct: Dict[str, bool] = {}
    for e in tqdm(all_eval):
        pred = generate_answer(e["prompt"], model, tokenizer, args.src_json, max_new=args.base_max_new)
        base_correct[e["prompt"]] = (e["gold"] in pred)

    if args.cache_base:
        try:
            cache_file.write_text(json.dumps(base_correct))
            print(f"[cache] Wrote base-correct ({len(base_correct)}) to {cache_file}")
        except Exception as e:
            print(f"[cache] Failed to write cache: {e}")

    return base_correct

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Sequential trainer with cached base-correct locality sampling."""

    def __init__(self, args, tokenizer, model, examples):
        self.args = args
        self.batch_k = args.batch_k
        self.tok = tokenizer
        self.model = model
        self.examples = examples
        self.device = model.device

        # Build logic → TRAIN/EVAL pools
        self.train_pool_by_logic: dict[str, list[dict]] = defaultdict(list)
        self.eval_pool_by_logic: dict[str, list[dict]] = defaultdict(list)
        for p in examples:
            self.train_pool_by_logic[p["logic"].lower()].append(p["train"])
            self.eval_pool_by_logic[p["logic"].lower()].append(p["eval"] | {"logic": p["logic"]})

        # Flat eval pool
        self.all_eval: List[dict] = []
        for buf in self.eval_pool_by_logic.values():
            self.all_eval.extend(buf)

        # Base-correct mask for locality sampling
        self.base_correct: Dict[str, bool] = load_or_build_base_correct(
            self.args, self.model, self.tok, self.all_eval
        )

        # Snapshot initial LoRA weights (only trainable parameters)
        self.orig_lora_state = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # Running accuracy counters
        self.gen_hits = 0
        self.gen_total = 0
        self.loc_hits = 0
        self.loc_total = 0
        self.edit_hits = 0
        self.edit_total = 0
        self.last_loss = 0.0

    def _sample_same_logic(self, logic: str, k: int, *, exclude_prompt: Optional[str] = None) -> List[dict]:
        pool = [e for e in self.eval_pool_by_logic[logic.lower()]]
        if exclude_prompt is not None:
            pool = [e for e in pool if e["prompt"] != exclude_prompt]
        if not pool:
            return []
        k = min(k, len(pool))
        return random.sample(pool, k=k)

    def _sample_locality(self, logic: str, k: int) -> List[dict]:
        candidates = [
            e for e in self.all_eval
            if e["logic"].lower() != logic.lower() and self.base_correct.get(e["prompt"], False)
        ]
        if not candidates:
            candidates = [e for e in self.all_eval if e["logic"].lower() != logic.lower()]
        if not candidates:
            candidates = list(self.all_eval)
        k = min(k, len(candidates))
        return random.sample(candidates, k=k)

    def run(self):
        print("\nStarting sequential training…")
        start = time.time()

        for step, pair in enumerate(self.examples, 1):
            self._reset_lora()
            self._process_one(pair)

            if step % 2 == 0 or step == len(self.examples):
                print(self._progress_str(step))

        elapsed = (time.time() - start) / 60.0
        print(f"Completed in {elapsed:.1f} min")

        self._save_results(self._make_outdir())

    def _reset_lora(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.orig_lora_state:
                    p.copy_(self.orig_lora_state[n])

    def _process_one(self, entry: dict):
        logic = entry["logic"]
        train_logic = logic.lower()

        # Fine-tune on same-logic batch
        if self.args.fine_tune:
            batch_examples = random.sample(
                self.train_pool_by_logic[train_logic],
                k=min(self.batch_k, len(self.train_pool_by_logic[train_logic]))
            )
            batch_tensors = {"input_ids": [], "attention_mask": [], "labels": []}
            for ex in batch_examples:
                enc = encode_example(ex, self.tok, self.args.src_json)
                for k in batch_tensors:
                    batch_tensors[k].append(enc[k])
            batch_stack = {k: torch.stack(v).to(self.device) for k, v in batch_tensors.items()}

            for _ in range(self.args.steps_per_batch):
                loss = self.model(**batch_stack).loss / len(batch_examples)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.last_loss = float(loss.detach().cpu())
        else:
            self.last_loss = 0.0

        # Evaluation
        # 1) Edited example itself (post-edit)
        edit_eval = entry["eval"] | {"logic": logic}
        pred_edit = generate_answer(edit_eval["prompt"], self.model, self.tok, self.args.src_json, self.args.base_max_new)
        self.edit_hits += int(edit_eval["gold"] in pred_edit)
        self.edit_total += 1

        # 2) SAME logic (exclude current prompt)
        gen_eval = self._sample_same_logic(logic, self.args.gen_k, exclude_prompt=entry["eval"]["prompt"])

        # 3) OTHER logic, base-correct only (locality)
        loc_eval = self._sample_locality(logic, self.args.loc_k)

        for e in gen_eval:
            pred = generate_answer(e["prompt"], self.model, self.tok, self.args.src_json, self.args.base_max_new)
            self.gen_hits += int(e["gold"] in pred)
            self.gen_total += 1

        for e in loc_eval:
            pred = generate_answer(e["prompt"], self.model, self.tok, self.args.src_json, self.args.base_max_new)
            self.loc_hits += int(e["gold"] in pred)
            self.loc_total += 1

    def _progress_str(self, step: int) -> str:
        e_acc = self.edit_hits / self.edit_total if self.edit_total else 0.0
        g_acc = self.gen_hits / self.gen_total if self.gen_total else 0.0
        l_acc = self.loc_hits / self.loc_total if self.loc_total else 0.0
        return (f"[{step:4d}/{len(self.examples)}]  "
                f"edit_acc={e_acc:.3f}  gen_acc={g_acc:.3f}  loc_acc={l_acc:.3f}  "
                f"loss={self.last_loss:.4f}")

    def _make_outdir(self) -> Path:
        lr_slug = str(self.args.lr).replace('.', 'p').replace('-', 'm')
        dataset_tag = Path(self.args.src_json).stem
        run_tag = Path(self.args.resume).stem if self.args.resume else "origin"
        out_dir = Path(self.args.out_root).resolve() / dataset_tag / lr_slug / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] Saving results under: {out_dir}")
        return out_dir

    def _save_results(self, out_dir: Path):
        e_acc = self.edit_hits / self.edit_total if self.edit_total else 0.0
        g_acc = self.gen_hits / self.gen_total if self.gen_total else 0.0
        l_acc = self.loc_hits / self.loc_total if self.loc_total else 0.0

        df = pd.DataFrame([{
            "edit_hits": self.edit_hits,
            "edit_total": self.edit_total,
            "edit_acc": round(e_acc, 6),
            "gen_hits": self.gen_hits,
            "gen_total": self.gen_total,
            "gen_acc": round(g_acc, 6),
            "loc_hits": self.loc_hits,
            "loc_total": self.loc_total,
            "loc_acc": round(l_acc, 6),
            "steps_per_batch": self.args.steps_per_batch,
            "batch_k": self.args.batch_k,
            "lr": self.args.lr,
            "model_name": self.args.model_name,
            "seed": self.args.seed,
        }])

        fname = f"summary_accuracy_seed{self.args.seed}"
        csv_path = out_dir / f"{fname}.csv"
        json_path = out_dir / f"{fname}.json"

        df.to_csv(csv_path, index=False)
        with json_path.open("w") as f:
            json.dump(df.iloc[0].to_dict(), f, indent=2)
        print(f"Saved accuracy summaries: {csv_path}  |  {json_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    # Guard: model_name must be a model id, not a directory
    if Path(args.model_name).exists() and Path(args.model_name).is_dir():
        raise ValueError(
            f"--model_name cannot be a directory: {args.model_name}\n"
            "Set --model_name to a base HF model id (e.g., Qwen/Qwen2.5-3B-Instruct)\n"
            "and pass your Bio ckpt directory or .pt file via --resume."
        )

    print(f"[cfg] model_name = {args.model_name}")
    print(f"[cfg] resume     = {args.resume}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(args.model_name)
    model = get_model(args.model_name, tokenizer, device, args)
    print("Loaded model with trainable LoRA parameters:")
    model.print_trainable_parameters()

    examples = harvest_examples(args.src_json, seed=args.seed)
    print(f"Loaded {len(examples)} examples (shuffled).")

    trainer = Trainer(args, tokenizer, model, examples)
    trainer.run()

if __name__ == "__main__":
    main()