#!/usr/bin/env python3
"""
AlphaEdit (single-file, fused)
---------------------------------
Self-contained implementation of an adapter-based (LoRA) editing loop with
"null-space protection" + prediction-protection for locality, tailored for your
logic datasets.

What it does
============
1) Loads a main PL dataset (e.g., data/logic/level_1.json) containing logic
   instances. For each step, it forms a same-logic mini-batch and performs a
   lightweight edit using LoRA.
2) Loads an augmented dataset (e.g., data/corrupt/augmented_dataset.json) that
   provides (clean prompt, answer) pairs per logic. This serves as a
   "protected" set to preserve base capabilities.
3) During each optimization step, the gradient update from the edit loss is
   explicitly orthogonalized w.r.t. a protected gradient direction estimated
   on the protected set (null-space protection). We also add a KL divergence
   to the base model’s predictions on the protected set (prediction protection).
4) Sequentially resets LoRA to the initial state for each edit, and evaluates
   edit accuracy, generality (same logic), and locality (other logics that were
   base-correct) with greedy decoding.

Key CLI flags
=============
--model_name           HF base model id (e.g., Qwen/Qwen2.5-3B-Instruct)
--src_json             Main logic json (e.g., data/logic/level_1.json)
--augmented_json       Augmented json for protected set (clean/corrupt/answers)
--lr                   Learning rate for LoRA params
--lambda_pred          Weight of prediction-protection KL on protected set
--beta_proj            Strength of null-space projection (0 disables projection)
--steps_per_batch      Inner steps for each same-logic batch edit
--batch_k              #train examples (same logic) per edit step
--gen_k, --loc_k       #eval samples for generality/locality per step

Output
======
Saves a CSV/JSON summary under outputs/perlogic/<dataset>/<lr>/<origin|ckpt>/
with edit/gen/local accuracies.

NOTE: No external project imports are required. Only needs standard libs +
transformers, peft, torch, numpy, pandas, tqdm.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# --------------------------- Constants & Paths ---------------------------
SEED = 10
MAX_LEN = 256
POSSIBLE_ANSWERS = ["true", "false", "n/a"]

try:
    from config.paths import DATA_DIR, OUTPUTS_DIR  # type: ignore
except Exception:  # noqa: E722
    DATA_DIR = Path("./data")
    OUTPUTS_DIR = Path("./outputs")

# --------------------------- Reproducibility ----------------------------
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------------- CLI ---------------------------------------

def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes"}:
        return True
    if v.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlphaEdit fused: LoRA editing with null-space + prediction protection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Datasets
    p.add_argument("--src_json", type=Path, default=DATA_DIR / "logic/level_1.json",
                   help="Main logic dataset (ContextHub-like)")
    p.add_argument("--augmented_json", type=Path,
                   default=DATA_DIR / "logic/level_1.json",
                   help="Augmented dataset providing protected (clean, answer)")

    # Model
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--resume", type=Path, default=None,
                   help="Optional full-model checkpoint to resume from")
    p.add_argument("--strict_resume", type=str2bool, default=False)

    # LoRA / Optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_k", type=int, default=1,
                   help="# of same-logic training examples per edit step")
    p.add_argument("--steps_per_batch", type=int, default=10)
    p.add_argument("--lambda_pred", type=float, default=1.0,
                   help="Weight of prediction-protection KL on protected set")
    p.add_argument("--beta_proj", type=float, default=1.0,
                   help="Scale for null-space projection (0 disables)")

    # Evaluation sampling
    p.add_argument("--gen_k", type=int, default=1)
    p.add_argument("--loc_k", type=int, default=1)
    p.add_argument("--base_max_new", type=int, default=5)

    # Protected loader config
    p.add_argument("--prot_max_len", type=int, default=256)
    p.add_argument("--prot_n_logics", type=int, default=16)
    p.add_argument("--prot_per_logic", type=int, default=2)
    p.add_argument("--prot_batch", type=int, default=4)

    # Base-correct locality cache
    p.add_argument("--cache_dir", type=Path, default=DATA_DIR / "correct")
    p.add_argument("--cache_base", type=str2bool, default=True)
    p.add_argument("--force_recompute_base", type=str2bool, default=False)

    # Misc
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out_root", type=Path, default=OUTPUTS_DIR / "perlogic")

    return p.parse_args()

# --------------------------- Tokenizer & Gen ----------------------------

def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode_supervised(prompt: str, answer: str, tok: AutoTokenizer) -> Dict[str, torch.Tensor]:
    full = prompt + " " + answer + tok.eos_token
    ids = tok(full, max_length=MAX_LEN, padding="max_length",
              truncation=True, return_tensors="pt")
    input_ids = ids["input_ids"].squeeze(0)
    attn_mask = ids["attention_mask"].squeeze(0)

    # mask prompt tokens out of loss
    prompt_len = tok(prompt, return_tensors="pt")["input_ids"].squeeze(0).numel()
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    labels[attn_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def format_prompt(nl: str) -> str:
    return f"{nl} (Answer only in True, False, or N/A (Neither)). Answer:"


def normalize_answer(text: str) -> str:
    a = text.strip().split()[0].lower().strip().strip(".,")
    if any(k in a for k in POSSIBLE_ANSWERS):
        return a
    low = text.lower()
    if any(x in low for x in {"yes", "true"}):
        return "true"
    if any(x in low for x in {"not", "no", "false"}):
        return "false"
    if "n/a" in low:
        return "n/a"
    return "true"


def greedy_answer(prompt: str, model, tok: AutoTokenizer, max_new: int) -> str:
    model.eval()
    with torch.no_grad():
        templ = format_prompt(prompt)
        ids = tok(templ, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
    try:
        tail = text.split(" Answer:")[-1]
    except Exception:
        tail = text
    return normalize_answer(tail)

# --------------------------- Data Loading -------------------------------

CUE = re.compile(
    r"\b(?:then|which implies|this (?:would )?implies?|would suggest that|"
    r"implies?|suggests? that)\b", re.I,
)


def load_main_examples(path: Path, seed: int) -> Tuple[List[dict], List[dict]]:
    """Load dataset -> (train_examples, eval_examples).
    Each example has keys: logic, train{text,label}, eval{prompt,gold}.
    """
    exs: list[dict] = []
    data = json.load(path.open())
    for rec in data:
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
                    "train": {"text": format_prompt(premise), "label": gold},
                    "eval": {"prompt": nl_full, "gold": gold},
                })
    rng = random.Random(seed)
    rng.shuffle(exs)

    train_pool_by_logic: dict[str, list[dict]] = defaultdict(list)
    eval_pool_by_logic: dict[str, list[dict]] = defaultdict(list)
    for p in exs:
        train_pool_by_logic[p["logic"].lower()].append(p["train"])
        eval_pool_by_logic[p["logic"].lower()].append(p["eval"] | {"logic": p["logic"]})

    all_eval = []
    for buf in eval_pool_by_logic.values():
        all_eval.extend(buf)

    return exs, all_eval


# Augmented JSON -> grouped rows per logic

def load_augmented_grouped(path: Path) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    with path.open() as fp:
        for block in json.load(fp):
            for prm in block.get("prompts", []):
                logic = prm.get("logic", "").strip()
                grouped[logic].append({
                    "clean": prm["clean"].strip(),
                    "answer": prm.get("answers", [""])[0].strip(),
                })
    return grouped


class ProtectedDataset(Dataset):
    """Protected set used for preservation.
    Builds tokenized prompts; last position of target_ids stores first token of answer.
    """
    def __init__(self,
                 grouped_rows: Dict[str, List[Dict]],
                 tokenizer: AutoTokenizer,
                 max_length: int = 256,
                 n_logics: int = 16,
                 per_logic: int = 2,
                 seed: int = 0):
        super().__init__()
        self.tok = tokenizer
        rng = random.Random(seed)
        all_logics = list(grouped_rows.keys())
        rng.shuffle(all_logics)
        picked = all_logics[:min(n_logics, len(all_logics))]
        pairs = []
        for lgc in picked:
            rows = grouped_rows[lgc]
            if not rows:
                continue
            idxs = list(range(len(rows)))
            rng.shuffle(idxs)
            for i in idxs[:per_logic]:
                r = rows[i]
                pairs.append((format_prompt(r["clean"]), r.get("answer", "")))
        self.N = len(pairs)
        enc = tokenizer([p for p, _ in pairs], padding="max_length",
                        truncation=True, max_length=max_length, return_tensors="pt")
        tgt_ids = []
        for _, ans in pairs:
            ids = tokenizer.encode(ans, add_special_tokens=False) or [tokenizer.eos_token_id]
            tgt_ids.append(ids[0])
        self.input_ids = enc["input_ids"]
        self.attn_mask = enc["attention_mask"]
        self.target_ids = torch.tensor(tgt_ids, dtype=torch.long)

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "target_id": self.target_ids[idx],
        }


def build_protected_loader(grouped_rows: Dict[str, List[Dict]],
                           tokenizer: AutoTokenizer,
                           max_length: int,
                           n_logics: int,
                           per_logic: int,
                           batch_size: int,
                           seed: int = 0):
    ds = ProtectedDataset(grouped_rows, tokenizer, max_length, n_logics, per_logic, seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)

# --------------------------- Models -------------------------------------

def _infer_ckpt_dtype(sd):
    for v in sd.values():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            return v.dtype
    return None


def _cast_state_dict(sd, dtype):
    out = {}
    for k, v in sd.items():
        out[k] = v.to(dtype) if torch.is_tensor(v) and torch.is_floating_point(v) else v
    return out


def load_models(model_name: str, args, tok: AutoTokenizer, device: torch.device):
    ckpt_sd = None
    ckpt_dtype = None
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        ckpt_sd = ckpt.get("model_state", ckpt)
        ckpt_dtype = _infer_ckpt_dtype(ckpt_sd)
    target_dtype = torch.bfloat16 if ckpt_dtype == torch.bfloat16 else torch.float16

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=target_dtype)
    base_model.to(device)
    base_model.eval()

    if ckpt_sd is not None:
        if ckpt_dtype is not None and ckpt_dtype != target_dtype:
            ckpt_sd = _cast_state_dict(ckpt_sd, target_dtype)
        missing, unexpected = base_model.load_state_dict(ckpt_sd, strict=args.strict_resume)
        print(f"Resumed base from {args.resume}, missing={len(missing)} unexpected={len(unexpected)}")

    # LoRA adapter on a *fresh* copy used for editing
    edit_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=target_dtype)

    target_modules = None
    if model_name.lower().startswith("google/gemma-3-4b-it"):
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules
    )
    edit_model = get_peft_model(edit_model, lora_cfg)
    edit_model.to(device)
    edit_model.generation_config.pad_token_id = tok.pad_token_id

    return base_model, edit_model

# --------------------------- Base-correct cache -------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def _cache_path(args: argparse.Namespace, src_file: Path) -> Path:
    dataset_tag = src_file.resolve().stem
    model_slug = _slug(args.model_name)
    ckpt_slug = _slug(args.resume.stem) if args.resume else "origin"
    key = f"{model_slug}_{ckpt_slug}"
    return Path(args.cache_dir) / dataset_tag / f"base_correct_{key}.json"


def load_or_build_base_correct(args, model, tok, all_eval: List[dict]) -> Dict[str, bool]:
    cache_file = _cache_path(args, args.src_json)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if args.cache_base and (not args.force_recompute_base) and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if isinstance(data, dict) and len(data) > 0:
                print(f"[cache] Loaded base-correct ({len(data)}) from {cache_file}")
                return {str(k): bool(v) for k, v in data.items()}
        except Exception as e:
            print(f"[cache] Failed to read cache ({e}), recomputing...")

    print("[cache] Computing base-correct set...")
    base_correct: Dict[str, bool] = {}
    for e in tqdm(all_eval):
        pred = greedy_answer(e["prompt"], model, tok, max_new=args.base_max_new)
        base_correct[e["prompt"]] = (e["gold"] in pred)

    if args.cache_base:
        try:
            cache_file.write_text(json.dumps(base_correct))
            print(f"[cache] Wrote base-correct ({len(base_correct)}) -> {cache_file}")
        except Exception as e:
            print(f"[cache] Failed to write cache: {e}")

    return base_correct

# --------------------------- AlphaEdit utils ----------------------------

def kl_to_base_on_last_token(edit_model, base_model, batch, tok: AutoTokenizer) -> torch.Tensor:
    """KL(edit || base) on the last position (target id provided)."""
    with torch.no_grad():
        base_logits = base_model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"]).logits[:, -1, :]
        base_prob = F.log_softmax(base_logits, dim=-1)  # log-prob of base
    edit_logits = edit_model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"]).logits[:, -1, :]
    edit_logprob = F.log_softmax(edit_logits, dim=-1)
    # Gather target token for readability (not used in KL), still compute KL over full vocab
    kl = F.kl_div(edit_logprob, base_prob.exp(), reduction="batchmean")
    return kl


def compute_protected_grad_direction(edit_model, batch) -> torch.Tensor:
    """Compute a single protected gradient vector g_p over LoRA trainable params.
    We backprop a CE loss on the target token (last position) for the protected batch,
    then flatten gradients of trainable (requires_grad) parameters and stack.
    """
    edit_model.zero_grad(set_to_none=True)
    out = edit_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits_last = out[:, -1, :]  # [B, V]
    loss = F.cross_entropy(logits_last, batch["target_id"].to(logits_last.device))
    loss.backward()

    grads = []
    for n, p in edit_model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().flatten())
    if not grads:
        return torch.zeros(1, device=out.device)
    g = torch.cat(grads)
    # Normalize to unit vector
    g_norm = torch.norm(g) + 1e-12
    return g / g_norm


def project_edit_gradients(edit_model, g_prot: torch.Tensor, beta: float = 1.0) -> None:
    """Project current gradients to be orthogonal to g_prot: g <- g - beta * proj_{g_prot}(g).
    If beta=0, no projection. In-place update on .grad of trainable params.
    """
    if beta <= 0 or g_prot.numel() == 1:
        return
    # Flatten current grads
    grads = []
    params = []
    for n, p in edit_model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.flatten())
            params.append(p)
    if not grads:
        return
    g = torch.cat(grads)
    # Projection: (g·u) u
    dot = torch.dot(g, g_prot)
    proj = dot * g_prot
    g_new = g - beta * proj

    # Scatter back
    offset = 0
    for p in params:
        sz = p.grad.numel()
        p.grad = g_new[offset:offset+sz].view_as(p.grad).detach().clone()
        offset += sz

# --------------------------- Trainer ------------------------------------
class AlphaEditTrainer:
    def __init__(self, args, tok, base_model, edit_model, examples, all_eval, prot_loader):
        self.args = args
        self.tok = tok
        self.base = base_model
        self.model = edit_model
        self.examples = examples
        self.all_eval = all_eval
        self.device = edit_model.device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        # Pools by logic
        self.train_by_logic: dict[str, list[dict]] = defaultdict(list)
        self.eval_by_logic: dict[str, list[dict]] = defaultdict(list)
        for p in examples:
            self.train_by_logic[p["logic"].lower()].append(p["train"])
            self.eval_by_logic[p["logic"].lower()].append(p["eval"] | {"logic": p["logic"]})

        # Base-correct set for locality
        self.base_correct = load_or_build_base_correct(args, base_model, tok, all_eval)

        # Save initial LoRA state
        self.init_lora = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

        # Metrics
        self.gen_hits = 0; self.gen_total = 0
        self.loc_hits = 0; self.loc_total = 0
        self.edit_hits = 0; self.edit_total = 0
        self.last_loss = 0.0

        # Protected loader (cycled)
        self.prot_loader = prot_loader
        self._prot_iter = iter(self.prot_loader) if self.prot_loader is not None else None

    def _prot_batch(self):
        if self._prot_iter is None:
            return None
        try:
            b = next(self._prot_iter)
        except StopIteration:
            self._prot_iter = iter(self.prot_loader)
            b = next(self._prot_iter)
        return {k: v.to(self.device) for k, v in b.items()}

    def _reset_lora(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.init_lora:
                    p.copy_(self.init_lora[n])

    def _sample_same_logic(self, logic: str, k: int, exclude_prompt: Optional[str] = None) -> List[dict]:
        pool = [e for e in self.eval_by_logic[logic.lower()]]
        if exclude_prompt is not None:
            pool = [e for e in pool if e["prompt"] != exclude_prompt]
        if not pool:
            return []
        return random.sample(pool, k=min(k, len(pool)))

    def _sample_locality(self, logic: str, k: int) -> List[dict]:
        cand = [e for e in self.all_eval if e["logic"].lower() != logic.lower() and self.base_correct.get(e["prompt"], False)]
        if not cand:
            cand = [e for e in self.all_eval if e["logic"].lower() != logic.lower()]
        if not cand:
            cand = list(self.all_eval)
        return random.sample(cand, k=min(k, len(cand)))

    def train_step(self, batch_examples: List[dict]):
        # Build supervised batch
        tensors = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in batch_examples:
            enc = encode_supervised(ex["text"], ex["label"], self.tok)
            for k in tensors:
                tensors[k].append(enc[k])
        batch = {k: torch.stack(v).to(self.device) for k, v in tensors.items()}

        # Optional protected batch
        prot = self._prot_batch()

        # 1) Compute protected gradient direction g_p (if available)
        g_p = None
        if prot is not None:
            g_p = compute_protected_grad_direction(self.model, prot)

        # 2) Edit steps with projection and prediction-protection
        loss_val = 0.0
        for _ in range(self.args.steps_per_batch):
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(**batch)
            loss_edit = out.loss
            loss = loss_edit
            if prot is not None and self.args.lambda_pred > 0:
                kl = kl_to_base_on_last_token(self.model, self.base, prot, self.tok)
                loss = loss + self.args.lambda_pred * kl
            loss.backward()
            if g_p is not None:
                project_edit_gradients(self.model, g_p, beta=self.args.beta_proj)
            self.optimizer.step()
            loss_val = float(loss.detach().cpu())
        self.last_loss = loss_val

    def evaluate_prompts(self, prompts: List[dict], max_new: int) -> Tuple[int, int]:
        hits = 0
        for e in prompts:
            pred = greedy_answer(e["prompt"], self.model, self.tok, max_new)
            hits += int(e["gold"] in pred)
        return hits, len(prompts)

    def run(self):
        print("\nStarting AlphaEdit sequential training…")
        start = time.time()
        for step, p in enumerate(self.examples, 1):
            self._reset_lora()
            logic = p["logic"]

            # Train on same-logic batch
            train_pool = self.train_by_logic[logic.lower()]
            batch = random.sample(train_pool, k=min(self.args.batch_k, len(train_pool)))
            self.train_step(batch)

            # Evaluate
            # (1) Edited example
            pred = greedy_answer(p["eval"]["prompt"], self.model, self.tok, self.args.base_max_new)
            self.edit_hits += int(p["eval"]["gold"] in pred)
            self.edit_total += 1

            # (2) Generality
            gen_eval = self._sample_same_logic(logic, self.args.gen_k, exclude_prompt=p["eval"]["prompt"]) \
                       if self.args.gen_k > 0 else []
            h, t = self.evaluate_prompts(gen_eval, self.args.base_max_new)
            self.gen_hits += h; self.gen_total += t

            # (3) Locality
            loc_eval = self._sample_locality(logic, self.args.loc_k) if self.args.loc_k > 0 else []
            h, t = self.evaluate_prompts(loc_eval, self.args.base_max_new)
            self.loc_hits += h; self.loc_total += t

            if step % 2 == 0 or step == len(self.examples):
                e_acc = self.edit_hits / max(1, self.edit_total)
                g_acc = self.gen_hits / max(1, self.gen_total)
                l_acc = self.loc_hits / max(1, self.loc_total)
                print(f"[{step:4d}/{len(self.examples)}]  edit_acc={e_acc:.3f}  gen_acc={g_acc:.3f}  "
                      f"loc_acc={l_acc:.3f}  loss={self.last_loss:.4f}")

        elapsed = (time.time() - start) / 60.0
        print(f"Completed in {elapsed:.1f} min")
        self.save_results(self._make_outdir())

    def _make_outdir(self) -> Path:
        lr_slug = str(self.args.lr).replace('.', 'p').replace('-', 'm')
        dataset_tag = Path(self.args.src_json).stem
        run_tag = Path(self.args.resume).stem if self.args.resume else "origin"
        out_dir = Path(self.args.out_root).resolve() / dataset_tag / lr_slug / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] Saving results under: {out_dir}")
        return out_dir

    def save_results(self, out_dir: Path):
        e_acc = self.edit_hits / max(1, self.edit_total)
        g_acc = self.gen_hits / max(1, self.gen_total)
        l_acc = self.loc_hits / max(1, self.loc_total)
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
            "lambda_pred": self.args.lambda_pred,
            "beta_proj": self.args.beta_proj,
            "model_name": self.args.model_name,
            "seed": self.args.seed,
        }])
        fname = f"alphaedit_summary_seed{self.args.seed}"
        (out_dir / f"{fname}.csv").write_text(df.to_csv(index=False))
        (out_dir / f"{fname}.json").write_text(json.dumps(df.iloc[0].to_dict(), indent=2))
        print(f"Saved accuracy summaries under {out_dir}")

# --------------------------- Main --------------------------------------

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = get_tokenizer(args.model_name)

    # Main + eval
    examples, all_eval = load_main_examples(args.src_json, seed=args.seed)
    print(f"Loaded {len(examples)} main examples (shuffled).")

    # Protected loader
    grouped_rows = load_augmented_grouped(args.augmented_json)
    prot_loader = build_protected_loader(
        grouped_rows, tok,
        max_length=args.prot_max_len,
        n_logics=args.prot_n_logics,
        per_logic=args.prot_per_logic,
        batch_size=args.prot_batch,
        seed=args.seed,
    ) if len(grouped_rows) > 0 else None

    # Models
    base_model, edit_model = load_models(args.model_name, args, tok, device)
    print("Loaded base and LoRA-editable models.")
    try:
        edit_model.print_trainable_parameters()
    except Exception:
        pass

    # Train
    trainer = AlphaEditTrainer(args, tok, base_model, edit_model, examples, all_eval, prot_loader)
    trainer.run()


if __name__ == "__main__":
    main()
