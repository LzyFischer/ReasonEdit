#!/usr/bin/env python3
from __future__ import annotations

###############################################################################
# Standard library imports
###############################################################################
import argparse
import json
import logging
import math
import random
import re
import pdb 
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

###############################################################################
# Third‑party imports
###############################################################################
import torch
import torch.nn.functional as F  # noqa: F401 (kept for future use)
from einops import reduce  # noqa: F401 (kept for future use)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm  # noqa: F401 (could be re‑enabled for prog‑bars)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

###############################################################################
# Torch backend configuration (<=2.0 API style)
###############################################################################
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info(
    "flash: %s  • mem‑eff: %s  • math: %s",
    torch.backends.cuda.flash_sdp_enabled(),
    torch.backends.cuda.mem_efficient_sdp_enabled(),
    torch.backends.cuda.math_sdp_enabled(),
)

SEED_DEFAULT = 42  # module‑level default for reproducibility

###############################################################################
# Helper utilities
###############################################################################

def set_seed(seed: int = SEED_DEFAULT) -> None:
    """Fully deterministic RNG seeding across ``random`` & ``torch``."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def topk_mask(x: torch.Tensor, frac: float) -> torch.Tensor:
    """Boolean mask selecting the top‑``frac`` proportion of |x| (≥ 1 element)."""
    k = max(1, math.ceil(frac * x.numel()))
    idx = torch.topk(x.abs(), k, largest=True).indices
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[idx] = True
    return mask


def to_device(batch: Dict, device: torch.device) -> Dict:
    """Recursively move any tensor values in *batch* to *device*."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def first_attr(obj, *candidates):
    """Return the first available attribute among *candidates* or raise."""
    for name in candidates:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{obj.__class__.__name__} lacks {candidates}")

###############################################################################
# Data‑loading helpers
###############################################################################
_ANSWER_TAIL = re.compile(r"\s*\(answer in .*?answer:\s*$", re.I | re.S)


def _canon(text: str) -> str:
    """Whitespace‑canonicalised, lower‑cased prompt with optional tail removed."""
    no_tail = _ANSWER_TAIL.sub("", text)
    no_guideline = no_tail.replace(" (Answer in True, False, or N/A (Neither)). Answer:", "")
    return " ".join(no_guideline.split()).lower()


def load_deductive_json(path: Path, keep_canon: bool = True) -> List[Dict]:
    """Parse nested deductive‑logic JSON into a flat list of rows."""
    rows: List[Dict] = []
    with path.open() as fp:
        for rec in json.load(fp):
            base_logic = rec["question"][0]["<nl>"].strip()
            gold_str = str(rec["answer"]).lower()
            for cat, cat_val in rec.items():
                if cat in {"question", "answer"} or not isinstance(cat_val, dict):
                    continue
                for detail in cat_val.values():
                    prompt = detail.get("<nl>")
                    if prompt is None:
                        continue
                    row = {"logic": base_logic, "prompt": prompt.strip() + " (Answer in True, False, or N/A (Neither)). Answer:", "gold": gold_str}
                    if keep_canon:
                        row["canon"] = _canon(prompt)
                    rows.append(row)
    return rows


def load_augmented_json(path: Path) -> List[Dict]:
    """Flatten augmented_dataset.json into a list of {clean, corrupt, answer}."""
    rows: List[Dict] = []
    with path.open() as fp:
        for block in json.load(fp):
            for prm in block.get("prompts", []):
                rows.append({
                    "clean": prm["clean"].strip() + " (Answer in True, False, or N/A (Neither)). Answer:",
                    "corrupt": prm["corrupt"].strip() + " (Answer in True, False, or N/A (Neither)). Answer:",
                    "answer": prm["answers"][0].strip(),
                })
    return rows

###############################################################################
# Residual‑stream attribution helpers
###############################################################################

def _register_hooks(model, ids, device, n_head: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Capture selected activations and return logits from a forward pass."""
    acts: Dict[str, torch.Tensor] = {}
    handles = []

    def add_hook(module, name):
        handles.append(module.register_forward_hook(lambda _m, _in, out: acts.__setitem__(name, out[0] if isinstance(out, tuple) else out)))

    for l, blk in enumerate(model.model.layers):
        add_hook(first_attr(blk, "input_layernorm", "ln_1"), f"resid_{l}")
        add_hook(first_attr(blk, "self_attn", "attn"), f"attn_{l}")
        add_hook(blk.mlp, f"mlp_{l}")

    logits = model(**ids, use_cache=False).logits

    for h in handles:
        h.remove()
    return acts, logits

def compute_edge_attr_batch(model, tok, pairs, device, n_head):
    model.eval()
    attributions = []
    for pair in pairs:
        attr = compute_edge_attr(model, tok, pair, device, n_head)
        attributions.append(attr)
    return torch.stack(attributions).mean(dim=0)  # average across batch


def compute_edge_attr(model, tok, pair, device, n_head: int) -> torch.Tensor:
    """
    Return a fixed-length attribution vector a[e] ∝ Σ_{b,t} ∂ℓ/∂h_e · Δh_e
    with   b = batch index,  t = token position.
    """
    model.train(False)                                   # disable dropout

    # Encode both prompts in one padded batch → identical seq length
    enc = tok([pair["clean"], pair["corrupt"]],
              return_tensors="pt", padding=True).to(device)
    ids, msk = enc["input_ids"], enc["attention_mask"]

    # ------------------------------------------------------------------ hooks
    acts_clean, acts_corrupt, handles = {}, {}, []

    def hook(store):
        def fn(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if torch.isnan(out).any():
                print(f"⚠️ NaN detected in hook: {module._name}")
            store[module._name] = out
            return output
        return fn

    def _arm_hooks(store):
        for i, blk in enumerate(model.model.layers):
            layer_dict = {
                "ln":   first_attr(blk, "input_layernorm", "ln_1"),
                "attn": first_attr(blk, "self_attn", "attn"),
                "mlp":  blk.mlp,
            }
            for short, mod in layer_dict.items():
                mod._name = f"{short}_{i}"                  # ➊ give every module a tag
                handles.append(mod.register_forward_hook(hook(store)))

    _arm_hooks(acts_clean)
    model(input_ids=ids[0:1], attention_mask=msk[0:1], use_cache=False)

    for h in handles:                                    # re-wire hooks
        h.remove()
    handles.clear()
    _arm_hooks(acts_corrupt)

    logits = model(input_ids=ids[1:2], attention_mask=msk[1:2],
                   use_cache=False).logits

    # ------------------------------------------------------------------ loss
    gold_logp = torch.log_softmax(logits[:, -1, :], dim=-1).max()
    grads = torch.autograd.grad(
        gold_logp, list(acts_corrupt.values()),
        create_graph=True, retain_graph=True
    )

    # ---------------------------------------------------------------- reduce
    contrib = []
    for g, name in zip(grads, acts_corrupt):
        h_c, h_k = acts_clean[name], acts_corrupt[name]
        # sum over batch & token axes *before* flattening
        if g.dim() == 4:                      # [B, T, n_head, D_h]
            a = (g * (h_c - h_k)).sum(dim=(0, 1, 3))      # → [n_head]
        elif g.dim() == 3:                    # [B, T, D]
            a = (g * (h_c - h_k)).sum(dim=(0, 1))         # → [D]
        else:
            raise ValueError(f"Unexpected rank for {name}: {g.shape}")

        contrib.append(a)

    return torch.cat(contrib)

###############################################################################
# Main pipeline class
###############################################################################
class ReasonEditPlus:
    """Encapsulates Phase‑I contrastive training and Phase‑II LoRA editing."""

    # ───────────────────── initialisation ─────────────────────
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model & tokenizer
        self.tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side="left")
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            # torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)

        # Data loading
        self.logic_rows = load_deductive_json(args.deductive_json)
        self.aug_rows = load_augmented_json(args.augmented_json)

        # Buckets for quick sampling
        self.bucket: Dict[str, List[Dict]] = defaultdict(list)
        for r in self.logic_rows:
            self.bucket[r["logic"]].append(r)

        self.target_pair = self.aug_rows[0]  # default LoRA target debugging only

        # LoRA configuration
        self.lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules.split(","),
        )

        # Build canonical‑prompt → logic map （connect logic and augmented rows）
        canon2logic = {r["canon"]: r["logic"] for r in self.logic_rows}
        self.aug_bucket: Dict[str, List[Dict]] = defaultdict(list)
        for row in self.aug_rows:
            lg = canon2logic.get(_canon(row["clean"]))
            if lg:
                row["logic"] = lg
                self.aug_bucket[lg].append(row)
            else:
                logger.warning("Augmented row unmatched to any logic: %s…", row["clean"][:60])

    # ───────────────── Phase I: helpers ─────────────────
    def _attr(self, pairs: List[Dict]) -> torch.Tensor:
        return compute_edge_attr_batch(
            self.model,
            self.tok,
            pairs,
            device=self.device,
            n_head=self.model.config.num_attention_heads,
        )

    # ───────────────── Phase I: contrastive enhancement ─────────────────
    def contrastive_enhance(self, sample_same: Callable[[], List[Dict]], sample_diff: Callable[[], List[Dict]]) -> None:
        """One epoch of Algorithm 1 (lines 3–11)."""
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.args.ce_lr)

        for step in range(self.args.ce_steps):
            B_minus = random.sample(sample_same(), self.args.batch_size)
            B_prime = random.sample(sample_same(), self.args.batch_size)
            B_diff  = random.sample(sample_diff(), self.args.batch_size)

            A, B, C = self._attr(B_minus), self._attr(B_prime), self._attr(B_diff)
            mask = topk_mask(A, self.args.tau) | topk_mask(B, self.args.tau) | topk_mask(C, self.args.tau)
            J = mask.nonzero(as_tuple=False).squeeze(-1)
            # J = torch.arange(mask.numel(), device=self.device)
            if J.numel() == 0:
                continue

            d_pos = (A[J] - B[J]).pow(2)
            d_neg = (A[J] - C[J]).pow(2)
            loss = d_pos.mean() - self.args.lambda_ce * d_neg.mean()

            opt.zero_grad()

            loss.backward()
            opt.step()
            logger.debug("[CE %d/%d] loss=%.4g", step + 1, self.args.ce_steps, loss.item())

    # ───────────────── Phase II: local LoRA edit ─────────────────
    def local_lora_edit_single(self, pair: Dict) -> None:
        """Insert LoRA adapters on selected modules & fine‑tune on *pair*."""
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.model = get_peft_model(self.model, self.lora_cfg)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.args.lora_lr)
        sched = get_linear_schedule_with_warmup(opt, 0, self.args.lora_steps)

        batch = self._encode(pair["clean"], pair["answer"])
        self.model.train()
        for _ in range(self.args.lora_steps):
            loss = self.model(**batch).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

    # ───────────────── Encoding helper ─────────────────
    def _encode(self, prompt: str, answer: str) -> Dict:
        full = f"{prompt} {answer}{self.tok.eos_token}"
        enc = self.tok(full, truncation=True, return_tensors="pt").to(self.device)
        ids, mask = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        labels = ids.clone(); labels[:-1] = -100  # supervise only the answer token
        return {"input_ids": ids.unsqueeze(0), "attention_mask": mask.unsqueeze(0), "labels": labels.unsqueeze(0)}

###############################################################################
# CLI helpers
###############################################################################

def get_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ReasonEdit+ (refactored)")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--deductive_json", type=Path, default=Path("data/logic/deductive_logic.json"))
    p.add_argument("--augmented_json", type=Path, default=Path("data/corrupt/augmented_dataset.json"))

    # Phase I (Contrastive)
    p.add_argument("--ce_steps", type=int, default=10)
    p.add_argument("--ce_lr", type=float, default=1e-10)
    p.add_argument("--lambda_ce", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1e-2)

    # Phase II (LoRA)
    p.add_argument("--lora_steps", type=int, default=30)
    p.add_argument("--lora_lr", type=float, default=1e-10)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--batch_size", type=int, default=4, help="Number of samples in each contrastive batch")

    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    return p

###############################################################################
# Main entry‑point
###############################################################################

def main() -> None:
    args = get_cli_parser().parse_args()
    set_seed(args.seed)

    editor = ReasonEditPlus(args)

    logger.info("Phase I ▶ contrastive circuit enhancement …")
    editor.contrastive_enhance(
        lambda: editor.aug_bucket[random.choice(list(editor.aug_bucket.keys()))],
        lambda: random.choice(editor.aug_rows),
    )

    logger.info("Phase II ▶ one‑shot local LoRA edit …")
    editor.local_lora_edit_single(editor.target_pair)
    logger.info("✓ Finished ReasonEdit+ demo.")

###############################################################################
if __name__ == "__main__":
    main()
