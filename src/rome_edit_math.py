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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: if you have these constants, import them; otherwise fall back to cwd
try:
    from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR  # noqa: F401
except Exception:
    DATA_DIR = Path("./data")
    OUTPUTS_DIR = Path("./outputs")


# ---------- Numeric parsing & compare ----------
NUM_RE = re.compile(r'[-+]?((\d{1,3}(,\d{3})+)|\d+)(\.\d+)?([eE][-+]?\d+)?')

def _parse_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace("−", "-")
    m = NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None

def _fmt_number(f: Optional[float]) -> str:
    if f is None:
        return ""
    return str(int(f)) if float(f).is_integer() else str(float(f))

def numeric_equal(pred: Optional[float], gold: Optional[float], *, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    if pred is None or gold is None:
        return False
    return abs(pred - gold) <= (atol + rtol * abs(gold))

###############################################################################
# 0. Global constants & reproducibility                                       #
###############################################################################
SEED = 10
MAX_LEN = 256
POSSIBLE_ANSWERS = ["true", "false", "n/a"]

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

###############################################################################
# 1. Configuration & CLI                                                      #
###############################################################################
def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fused ROME editor (single-file) for logic dataset; optional LoRA FT; cached base-correct locality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # training behaviour
    parser.add_argument("--fine_tune", type=str2bool, default=False, help="Also do small LoRA FT before evaluation")
    parser.add_argument("--lr", type=float, default=1e-10, help="Learning rate for the LoRA parameters")
    parser.add_argument("--batch_k", type=int, default=1, help="# of same-logic training examples per step")

    # dataset paths
    parser.add_argument("--src_json", type=Path, default=DATA_DIR / "logic/math.json")

    # sampling-scheme (evaluation)
    parser.add_argument("--gen_k", type=int, default=1, help="# SAME-logic eval prompts per step")
    parser.add_argument("--loc_k", type=int, default=1, help="# locality eval prompts (base-correct, other logics)")

    # model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")

    # ckpt
    parser.add_argument("--resume", type=Path, default=None, help="Path to full-model checkpoint (optional)")
    parser.add_argument("--strict_resume", type=str2bool, default=False, help="strict load_state_dict")

    # output
    parser.add_argument("--out_root", type=Path, default=OUTPUTS_DIR / "perlogic_rome", help="Base dir to save results")

    # misc
    parser.add_argument("--steps_per_batch", type=int, default=10, help="Gradient steps per same-logic mini-batch")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for shuffling and sampling")

    # cache for base-correct locality pool
    parser.add_argument("--cache_dir", type=Path, default=DATA_DIR / "correct", help="Dir for base-correct cache")
    parser.add_argument("--cache_base", type=str2bool, default=True, help="Use/write base-correct cache")
    parser.add_argument("--force_recompute_base", type=str2bool, default=False, help="Ignore cache and recompute")
    parser.add_argument("--base_max_new", type=int, default=6, help="max_new_tokens for base-correct inference")

    # ROME controls
    parser.add_argument("--use_rome", type=str2bool, default=True, help="Apply ROME-style rank-1 edit per step")
    parser.add_argument("--edit_layer", type=int, default=-10, help="Which transformer block to edit (int or negative index)")
    parser.add_argument("--rome_alpha", type=float, default=1.0, help="Edit strength along target residual direction")
    parser.add_argument("--rome_l2", type=float, default=0.0001, help="Ridge lambda in closed-form update")
    parser.add_argument("--rome_dryrun", type=str2bool, default=False, help="Do not actually modify weights (for debug)")

    return parser.parse_args()

###############################################################################
# 2. Tokenizer, encoding & generation helpers                                 #
###############################################################################
def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def _answer_prompt_text(prompt: str) -> str:
    # 改为数值题专用提示
    return f"{prompt}\nAnswer with only the final numeric result.\nAnswer:"

def _norm_answer(ans: str) -> str:
    a = ans.lower().strip().strip(".").strip(",")
    if "true" in a or a in {"t", "yes"}:
        return "true"
    if "false" in a or a in {"f", "no", "not"}:
        return "false"
    if "n/a" in a or "neither" in a or "na" == a:
        return "n/a"
    return a

def generate_answer(prompt: str, model, tokenizer, src_path: Path, max_new: int = 16) -> Optional[float]:
    """生成后从 Answer: 之后解析数值（float），失败返回 None。"""
    model.eval()
    with torch.no_grad():
        templ = _answer_prompt_text(prompt)
        ids = tokenizer(templ, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    return _parse_float(text.strip())

###############################################################################
# 3. Data preparation                                                         #
###############################################################################
CUE = re.compile(
    r"\b(?:then|which implies|this (?:would )?implies?|would suggest that|"
    r"implies?|suggests? that)\b", re.I,
)

def harvest_examples_from_numeric(path: Path, seed: int) -> List[dict]:
    """
    支持两种根格式：
      A) dict: {template_id: [ {"problem","result"}, ... ]}
      B) 旧逻辑格式 list[...]（原脚本已有），这里保留兼容
    输出统一为：
      { "logic": <tid或逻辑名>,
        "train": {"text": <prompt_for_train>, "label": <gold_str>},
        "eval":  {"prompt": <prompt_for_eval>, "gold":  <gold_str>} }
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    exs: List[dict] = []

    # A) 你的新格式
    if isinstance(raw, dict):
        for tid, arr in raw.items():
            if not isinstance(arr, list):
                continue
            for s in arr:
                prob = str(s.get("problem", "")).strip()
                ans  = str(s.get("result", "")).strip()
                exs.append({
                    "logic": str(tid),
                    # 训练时把“数值答案”也拼到末尾监督（encode_example 里会处理）
                    "train": {"text": prob, "label": ans},
                    # 评测就用同一个 problem（如果你有 corrupt，可在此替换为 corrupt）
                    "eval":  {"prompt": prob, "gold": ans},
                })

    # B) 旧格式兼容（布尔逻辑数据），如果仍需支持的话
    elif isinstance(raw, list):
        for rec in raw:
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
    else:
        raise ValueError(f"Unsupported json root type: {type(raw)}")

    rng = random.Random(seed)
    rng.shuffle(exs)
    return exs

###############################################################################
# 4. Model & LoRA initialisation                                              #
###############################################################################
def _infer_ckpt_dtype(sd):
    for v in sd.values():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            return v.dtype
    return None

def _cast_state_dict(sd, dtype):
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            out[k] = v.to(dtype)
        else:
            out[k] = v
    return out

def get_model(model_name: str, tokenizer, device: torch.device, args):
    ckpt_sd = None
    ckpt_dtype = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        ckpt_sd = ckpt.get("model_state", ckpt)
        ckpt_dtype = _infer_ckpt_dtype(ckpt_sd)

    # dtype
    target_dtype = torch.bfloat16 if ckpt_dtype == torch.bfloat16 else torch.float16

    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=target_dtype)

    if ckpt_sd is not None:
        if ckpt_dtype is not None and ckpt_dtype != target_dtype:
            ckpt_sd = _cast_state_dict(ckpt_sd, target_dtype)
        missing, unexpected = base.load_state_dict(ckpt_sd, strict=args.strict_resume)
        print(f"Resumed from {args.resume}, missing={len(missing)} unexpected={len(unexpected)}")

    # LoRA (optional)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=None
    )
    model = get_peft_model(base, lora_cfg).to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model

###############################################################################
# 5. Base-correct cache helpers                                               #
###############################################################################
def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)

def _base_cache_path(args: argparse.Namespace, src_file: Path) -> Path:
    dataset_tag = src_file.resolve().stem
    model_slug = _slug(args.model_name)
    ckpt_slug = _slug(args.resume.stem) if args.resume else "origin"
    key = f"{model_slug}_{ckpt_slug}_maxnew{args.base_max_new}"
    return Path(args.cache_dir) / dataset_tag / f"base_correct_{key}.json"

@torch.no_grad()
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

    print("[cache] Computing base-correct set...")
    base_correct: Dict[str, bool] = {}
    for e in all_eval:
        pred = generate_answer(e["prompt"], model, tokenizer, args.src_json, max_new=args.base_max_new)
        ok = numeric_equal(pred, _parse_float(e["gold"]), atol=1e-6, rtol=1e-6)
        base_correct[e["prompt"]] = bool(ok)

    if args.cache_base:
        try:
            cache_file.write_text(json.dumps(base_correct))
            print(f"[cache] Wrote base-correct ({len(base_correct)}) to {cache_file}")
        except Exception as e:
            print(f"[cache] Failed to write cache: {e}")

    return base_correct

###############################################################################
# 6. ROME-style single-file editor                                            #
###############################################################################
class SimpleROME:
    """
    Minimal ROME-style editor:
      - Choose a transformer block (args.edit_layer).
      - Capture the MLP 'key' vector a = SiLU(up(h)) * sigmoid(gate(h)) at the answer position.
      - Choose a target residual delta Δr to boost target token vs ref token.
      - Do a rank-1 update on down_proj: ΔW = Δr ⊗ a / (||a||^2 + λ).
    Works with Qwen2-style SwiGLU MLPs (up_proj, gate_proj, down_proj).
    """

    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tok = tokenizer
        self.args = args
        # locate blocks
        # supports typical HF naming: model.model.layers[i].mlp
        self.blocks = model.get_base_model().model.layers
        self.edit_idx = args.edit_layer if args.edit_layer >= 0 else (len(self.blocks) + args.edit_layer)
        if not (0 <= self.edit_idx < len(self.blocks)):
            raise ValueError(f"edit_layer out of range: {args.edit_layer} (num_layers={len(self.blocks)})")
        self.block = self.blocks[self.edit_idx]
        # mlp parts
        self.mlp = self.block.mlp
        # check attributes
        assert hasattr(self.mlp, "up_proj") and hasattr(self.mlp, "gate_proj") and hasattr(self.mlp, "down_proj"), \
            "Expected SwiGLU MLP with up_proj, gate_proj, down_proj"

        # lm_head (unembedding)
        self.unembed = self.model.get_output_embeddings().weight  # [V, d_model]

        # answer token ids
        self.ans_ids = {k: self._first_token_id(k) for k in POSSIBLE_ANSWERS}

    def _first_token_id(self, word: str) -> int:
        ids = self.tok(word, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) > 0 else self.tok.convert_tokens_to_ids(word)

    @torch.no_grad()
    def _capture_key_vec(self, prompt_text: str) -> torch.Tensor:
        """
        Run a forward pass and capture the MLP 'key' vector a at the last
        position of the prompt (i.e., the 'Answer:' position).
        """
        # Build input up to Answer:
        full = _answer_prompt_text(prompt_text)
        enc = self.tok(full, return_tensors="pt").to(self.model.device)
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        pos = int(attn_mask[0].sum().item()) - 1  # last prompt token index

        # We need the block's pre-MLP hidden h at 'pos'
        # We'll directly recompute a = SiLU(up(h)) * sigmoid(gate(h)), using the block's MLP
        # To get h at each layer, run the base model with output_hidden_states=True
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        # hidden_states is a tuple: [layer0_out, layer1_out, ..., layerN_out] each [B, T, d_model]
        hs_tuple = outputs.hidden_states  # includes embeddings as 0? For Qwen2, hidden_states[0] is embeddings output
        # The block output at layer edit_idx corresponds to hidden_states[edit_idx+1]
        # (because hs[0] is the embeddings output). Confirm this by convention:
        h = hs_tuple[self.edit_idx + 1][0, pos]  # [d_model]

        # MLP key vector a (post activation, pre-down_proj)
        # SwiGLU: act(up(h)) * sigmoid(gate(h))
        up = self.mlp.up_proj(h)
        gate = self.mlp.gate_proj(h)
        a = F.silu(up) * torch.sigmoid(gate)  # [d_ff]
        return a  # torch.float16/bfloat16

    @torch.no_grad()
    def _target_residual(self, target_word: str, ref_word: Optional[str] = None) -> torch.Tensor:
        """
        Build residual direction Δr to push target logit up vs ref.
        Use unembedding row difference of target and ref.
        """
        vt = self.unembed[self.ans_ids[target_word]]  # [d_model]
        if ref_word is None:
            # take an average of competing answers (others)
            others = [w for w in POSSIBLE_ANSWERS if w != target_word]
            vr = torch.stack([self.unembed[self.ans_ids[w]] for w in others], dim=0).mean(dim=0)
        else:
            vr = self.unembed[self.ans_ids[ref_word]]
        direction = vt - vr
        direction = direction / (direction.norm(p=2) + 1e-6)
        return self.args.rome_alpha * direction  # [d_model]

    @torch.no_grad()
    def edit_once(self, prompt_text: str, desired_label: str):
        """
        Perform one ROME rank-1 update on down_proj weights at edit layer,
        using the captured key vector at 'Answer:' and desired label.
        """
        desired_label = _norm_answer(desired_label)
        if desired_label not in POSSIBLE_ANSWERS:
            # default to 'true'
            desired_label = "true"

        # capture key a
        a = self._capture_key_vec(prompt_text)  # [d_ff]
        a_dtype = self.mlp.down_proj.weight.dtype
        a = a.to(self.mlp.down_proj.weight.dtype)

        # choose ref as current "best competitor" among the remaining answers via embedding similarity (optional)
        # simple heuristic: pick the farthest
        others = [w for w in POSSIBLE_ANSWERS if w != desired_label]
        ref = None  # let _target_residual average the others

        # desired residual change Δr in model hidden dim
        delta_r = self._target_residual(desired_label, ref_word=ref).to(a_dtype)  # [d_model]

        # rank-1 closed-form for down_proj: ΔW = Δr ⊗ a / (||a||^2 + λ)
        denom = (a @ a) + self.args.rome_l2
        outer = torch.ger(delta_r, a) / denom.clamp_min(1e-6)  # [d_model, d_ff]

        if not self.args.rome_dryrun:
            self.mlp.down_proj.weight.add_(outer)  # in-place edit

###############################################################################
# 7. Trainer                                                                  #
###############################################################################
class Trainer:
    """Sequential trainer with cached base-correct locality sampling + ROME editor."""

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

        # ---------- Load or build cached base-correct ----------
        self.base_correct: Dict[str, bool] = load_or_build_base_correct(
            self.args, self.model, self.tok, self.all_eval
        )

        # Snapshot initial LoRA weights (only trainable parameters)
        self.orig_lora_state = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # ROME editor
        self.rome = SimpleROME(self.model, self.tok, self.args) if self.args.use_rome else None

        # Running accuracy counters
        self.gen_hits = 0
        self.gen_total = 0
        self.loc_hits = 0
        self.loc_total = 0
        self.edit_hits = 0
        self.edit_total = 0
        self.last_loss = 0.0

    # -------------------------- sampling helpers --------------------------
    def _sample_same_logic(self, logic: str, k: int, *, exclude_prompt: Optional[str] = None) -> List[dict]:
        pool = [e for e in self.eval_pool_by_logic[logic.lower()]]
        if exclude_prompt is not None:
            pool = [e for e in pool if e["prompt"] != exclude_prompt]
        if not pool:
            return []
        k = min(k, len(pool))
        return random.sample(pool, k=k)

    def _sample_locality(self, logic: str, k: int) -> List[dict]:
        """Sample OTHER-logic eval prompts that were base-correct; fallback to all OTHER-logic if empty."""
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

    # -------------------------- core loop ---------------------------------
    def run(self):
        print("\nStarting sequential ROME editing…")
        start = time.time()

        for step, pair in enumerate(self.examples, 1):
            # Reset LoRA to initial state (fresh adapter each step)
            self._reset_lora()

            # --------- (A) Optional: small LoRA FT on same-logic batch ---------
            if self.args.fine_tune:
                self._lora_ft_one(pair)

            # --------- (B) ROME single-step rank-1 edit on this sample ---------
            if self.rome is not None:
                prompt_for_key = pair["train"]["text"]
                desired = pair["train"]["label"]
                self.rome.edit_once(prompt_for_key, desired)

            # --------- (C) Evaluate ---------
            self._eval_one(pair)

            # Progress log
            if step % 2 == 0 or step == len(self.examples):
                print(self._progress_str(step))

        elapsed = (time.time() - start) / 60.0
        print(f"Completed in {elapsed:.1f} min")

        # Save aggregated metrics (simple CSV/JSON)
        self._save_results(self._make_outdir())

    def _reset_lora(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.orig_lora_state:
                    p.copy_(self.orig_lora_state[n])

    def _lora_ft_one(self, entry: dict):
        train_logic = entry["logic"].lower()
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
            loss = self.model(**batch_stack).loss / max(1, len(batch_examples))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.last_loss = float(loss.detach().cpu())

    def _eval_one(self, entry: dict):
        logic = entry["logic"]

        # 1) Edited example itself (post-edit accuracy)
        edit_eval = entry["eval"] | {"logic": logic}
        pred_edit = generate_answer(edit_eval["prompt"], self.model, self.tok, self.args.src_json, self.args.base_max_new)
        self.edit_hits += int(numeric_equal(pred_edit, _parse_float(edit_eval["gold"]), atol=1e-6, rtol=1e-6))
        self.edit_total += 1

        # 2) Generality: SAME logic (exclude current prompt)
        gen_eval = self._sample_same_logic(logic, self.args.gen_k, exclude_prompt=entry["eval"]["prompt"])

        # 3) Locality: ONLY from base-correct OTHER-logic samples
        loc_eval = self._sample_locality(logic, self.args.loc_k)

        for e in gen_eval:
            pred = generate_answer(e["prompt"], self.model, self.tok, self.args.src_json, self.args.base_max_new)
            self.gen_hits += int(numeric_equal(pred, _parse_float(e["gold"]), atol=1e-6, rtol=1e-6))
            self.gen_total += 1

        for e in loc_eval:
            pred = generate_answer(e["prompt"], self.model, self.tok, self.args.src_json, self.args.base_max_new)
            self.loc_hits += int(numeric_equal(pred, _parse_float(e["gold"]), atol=1e-6, rtol=1e-6))
            self.loc_total += 1

    def _progress_str(self, step: int) -> str:
        e_acc = self.edit_hits / self.edit_total if self.edit_total else 0.0
        g_acc = self.gen_hits / self.gen_total if self.gen_total else 0.0
        l_acc = self.loc_hits / self.loc_total if self.loc_total else 0.0
        return (f"[{step:4d}/{len(self.examples)}]  "
                f"edit_acc={e_acc:.3f}  gen_acc={g_acc:.3f}  loc_acc={l_acc:.3f}  "
                f"loss={self.last_loss:.4f}")

    # -------------------------- outputs -----------------------------------
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
            "use_rome": self.args.use_rome,
            "edit_layer": self.args.edit_layer,
            "rome_alpha": self.args.rome_alpha,
            "rome_l2": self.args.rome_l2,
        }])

        fname = f"summary_accuracy_seed{self.args.seed}"
        csv_path = out_dir / f"{fname}.csv"
        json_path = out_dir / f"{fname}.json"

        df.to_csv(csv_path, index=False)
        with json_path.open("w") as f:
            json.dump(df.iloc[0].to_dict(), f, indent=2)
        print(f"Saved accuracy summaries: {csv_path}  |  {json_path}")

###############################################################################
# 8. Supervised encoding (kept from your original)                            #
###############################################################################
def encode_example(row: dict, tokenizer, src_path: Path):
    # 数值题提示
    prompt = f"{row['text']}\nAnswer with only the final numeric result.\nAnswer:"
    gold = _parse_float(row["label"])
    ans_str = _fmt_number(gold)

    full = prompt + " " + ans_str + tokenizer.eos_token
    ids = tokenizer(full, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    input_ids = ids["input_ids"].squeeze(0)
    attn_mask = ids["attention_mask"].squeeze(0)
    labels = input_ids.clone()

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0).numel()
    labels[:prompt_len] = -100
    labels[attn_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

###############################################################################
# 9. Main                                                                     #
###############################################################################
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(args.model_name)
    model = get_model(args.model_name, tokenizer, device, args)
    print("Loaded model with trainable LoRA parameters (LoRA may be unused if --fine_tune false):")
    model.print_trainable_parameters()

    examples = harvest_examples_from_numeric(args.src_json, seed=args.seed)
    print(f"Loaded {len(examples)} examples (shuffled).")

    trainer = Trainer(args, tokenizer, model, examples)
    trainer.run()

if __name__ == "__main__":
    main()