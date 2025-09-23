#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contrastive_reptile.py (null-space protection removed)

Train a causal-LM with **Reptile meta-learning** on a contrastive logic dataset.
Two contrastive losses are treated as two meta-tasks (Task-A / Task-B).

Changes in this version:
• Full argparse config (no hard-coded constants)
• DATASET_TAG = Path(--data_json).stem  (e.g., data/corrupt/level_1.json → "level_1")
• Checkpoints saved under: ckpts/<DATASET_TAG>/<SAVE_TAG>_{step}.pt
• All null-space/Oja/functional projector code removed
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import trange

# ───────────────────────── Project-local helpers ─────────────────────────
from src.get_dataset import (
    LogicDataset,
    load_augmented_json_grouped,
    collate_fn,
)
import src.attribute_patch as AP  # exposes DEVICE / DTYPE / MODEL_NAME / AutoTokenizer / AutoModelForCausalLM / ActCacher / calculate_effect


# ═══════════════════════ Argparse / dataset tag ═══════════════════════

def dataset_tag_from_path(p: Path) -> str:
    """Return just the file stem as the dataset tag."""
    return Path(p).with_suffix("").name

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Reptile meta-learning on a contrastive logic dataset (no null-space protection)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # data / model
    ap.add_argument("--data_json", type=Path, default=Path("data/corrupt/level_1.json"))
    ap.add_argument("--model_name", type=str, default=None, help="Override AP.MODEL_NAME if set")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--group_size", type=int, default=2)
    ap.add_argument("--n_logic_per_item", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    # reptile / inner loop
    ap.add_argument("--meta_iters", type=int, default=45)
    ap.add_argument("--inner_steps", type=int, default=5)
    ap.add_argument("--inner_lr", type=float, default=1e-6)
    ap.add_argument("--meta_lr", type=float, default=1e-6)
    ap.add_argument("--tasks_per_meta", type=int, default=2)

    # saving
    ap.add_argument("--ckpt_root", type=Path, default=Path("ckpts"))
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--save_tag", type=str, default="reptile_d")
    ap.add_argument("--dataset_tag", type=str, default=None,
                    help="If set, overrides the default (Path(data_json).stem)")

    # consistency loss
    ap.add_argument("--consistency_lambda", type=float, default=0.1,
                    help="Weight of effect-consistency penalty")
    ap.add_argument("--consistency_normed", type=int, choices=[0, 1], default=1,
                    help="1=cosine distance on L2-normalized effects; 0=MSE")

    args = ap.parse_args()

    # derive dataset tag & ckpt dir
    tag = args.dataset_tag or dataset_tag_from_path(args.data_json)
    args.dataset_tag = tag
    args.ckpt_dir = (args.ckpt_root / tag)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    return args

args = get_args()

# ───────────────────────── Globals from AP + args ────────────────────────
DEVICE = AP.DEVICE
DTYPE = AP.DTYPE
MODEL_NAME = args.model_name or AP.MODEL_NAME


# ═══════════════════════ Distribution consistency mem ═══════════════════════

class DistConsistencyMemory:
    """
    Store previous-epoch flattened *effect* vectors for each (logic, sample-signature).
    During current epoch, penalize deviation from previous epoch on the SAME sample.
    """
    def __init__(self, lam: float = 0.1, normed: bool = True):
        self.prev: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)  # logic -> sig -> e_prev(float16, CPU)
        self.curr: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.lam = float(lam)
        self.normed = bool(normed)

    @staticmethod
    def _sig_from_ids(clean_ids: torch.Tensor, corrupt_ids: torch.Tensor) -> str:
        # short BLAKE2b signature of token id tensors
        h = hashlib.blake2b(digest_size=12)
        h.update(clean_ids.detach().to(torch.int32).contiguous().cpu().numpy().tobytes())
        h.update(corrupt_ids.detach().to(torch.int32).contiguous().cpu().numpy().tobytes())
        return h.hexdigest()

    def on_epoch_end(self):
        self.prev = self.curr
        self.curr = defaultdict(dict)

    def update_curr(self, logic: str, sig: str, e_now: torch.Tensor):
        self.curr[logic][sig] = e_now.detach().to("cpu", torch.float16).clone()

    def penalty(self, logic: str, sig: str, e_now: torch.Tensor) -> torch.Tensor:
        """
        Return scalar penalty for current effect vs previous epoch's.
        Cosine distance if normed=True; else MSE.
        """
        e_prev_cpu = self.prev.get(logic, {}).get(sig, None)
        if e_prev_cpu is None:
            return e_now.new_zeros(())
        e_prev = e_prev_cpu.to(e_now.device, e_now.dtype)
        if self.normed:
            e1 = nn.functional.normalize(e_now, dim=0, eps=1e-8)
            e0 = nn.functional.normalize(e_prev, dim=0, eps=1e-8)
            cos = torch.sum(e1 * e0)
            return (1.0 - cos)
        else:
            return nn.functional.mse_loss(e_now, e_prev, reduction="mean")


EFFECT_MEM = DistConsistencyMemory(
    lam=args.consistency_lambda,
    normed=bool(args.consistency_normed)
)


# ═══════════════════════ Utility: params clone / load ═══════════════════════

def clone_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def load_params_(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    for k, v in model.state_dict().items():
        v.data.copy_(state_dict[k].to(v.device, non_blocking=True))


# ═══════════════════════ Cache housekeeping ═══════════════════════

def purge_cache(cache: AP.ActCacher):
    for k, t in cache.cache.items():
        if t is not None:
            t.grad = None
        cache.cache[k] = None
    cache.cache.clear()


# ═══════════════════════ Checkpoint helper (dataset-tagged) ═══════════════════════

def save_checkpoint(step: int, model: nn.Module, ckpt_dir: Path, tag: str) -> Path:
    fname = f"{tag}_{step:05d}.pt"
    path = ckpt_dir / fname
    payload = {
        "iter": step,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "MODEL_NAME": MODEL_NAME,
            "GROUP_SIZE": args.group_size,
            "N_LOGIC_PER_ITEM": args.n_logic_per_item,
            "MAX_LEN": args.max_len,
            "BATCH_SIZE": args.batch_size,
            "SEED": args.seed,
            "META_ITERS": args.meta_iters,
            "INNER_STEPS": args.inner_steps,
            "INNER_LR": args.inner_lr,
            "META_LR": args.meta_lr,
            "TASKS_PER_META": args.tasks_per_meta,
            "DTYPE": str(DTYPE),
            "DATA_JSON": str(args.data_json),
            "DATASET_TAG": args.dataset_tag,
        },
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "rng_state_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            payload["rng_state_cuda"] = torch.cuda.get_rng_state()
        except Exception:
            pass
    torch.save(payload, path)
    torch.save(payload, ckpt_dir / f"{tag}_latest.pt")
    return path


# ═══════════════════════ Effect flattening helpers ═══════════════════════

def flatten_effect_dict(effect_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    flat = [v.flatten() for _, v in sorted(effect_dict.items())]
    return torch.cat(flat, 0)

def flatten_effects_to_embeddings(effects: Dict[str, List[Dict[str, torch.Tensor]]]) -> Dict[str, List[torch.Tensor]]:
    flattened: Dict[str, List[torch.Tensor]] = {}
    for logic, effect_dicts in effects.items():
        flattened.setdefault(logic, [])
        for eff in effect_dicts:
            flat_parts = [t.flatten() for _, t in sorted(eff.items())]
            flattened[logic].append(torch.cat(flat_parts, 0))
    return flattened


# ═══════════════════════ Build model / dataset / loader ═══════════════════════

print("[INFO] Loading model …")
tok = AP.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AP.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    attn_implementation="eager",
    device_map=None,
).to(AP.DEVICE)
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None
model.gradient_checkpointing_enable()

rows = load_augmented_json_grouped(args.data_json)
print(f"[INFO] Loaded {len(rows)} rows → grouping …")

ds = LogicDataset(
    data=rows,
    tokenizer=tok,
    group_size=args.group_size,
    n_logic_per_item=args.n_logic_per_item,
    max_length=args.max_len,
    seed=args.seed,
)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
loader_iter = iter(loader)
epoch_idx = 0
seen_in_epoch = 0

nodes = AP.get_comp_nodes(model)
print(f"[INFO] Tracking {len(nodes)} computational nodes")


# ═══════════════════════ Task loss (contrastive + consistency) ═══════════════════════

def compute_task_loss(item: Dict, task_idx: int) -> torch.Tensor:
    """
    Returns: contrastive_loss + args.consistency_lambda * mean_sample_consistency
    Effect-consistency compares each current sample's effect to its previous-epoch effect.
    """
    effects: Dict[str, List[Dict[str, torch.Tensor]]] = {}

    consistency_sum = None
    consistency_count = 0

    # Forward over group
    for logic, pair_list in item.items():
        effects[logic] = []
        for g1_dict in pair_list[0]:
            clean_ids = g1_dict["clean_ids"].to(AP.DEVICE)
            clean_mask = g1_dict["clean_mask"].to(AP.DEVICE)
            corrupt_ids = g1_dict["corrupt_ids"].to(AP.DEVICE)
            corrupt_mask = g1_dict["corrupt_mask"].to(AP.DEVICE)
            answers = g1_dict["answers_clean"]

            inputs_clean = {"input_ids": clean_ids, "attention_mask": clean_mask}
            inputs_cor = {"input_ids": corrupt_ids, "attention_mask": corrupt_mask}

            clean_cache = AP.ActCacher(model, nodes)
            corrupt_cache = AP.ActCacher(model, nodes)
            with clean_cache:
                out_clean = model(**inputs_clean)
            with corrupt_cache:
                out_corrupt = model(**inputs_cor)

            effect = AP.calculate_effect(
                model, clean_cache, corrupt_cache, nodes, tok, out_clean, answers,
            )
            effects[logic].append(effect)

            # Distribution constraint on the SAME sample vs previous epoch
            e_now_flat = flatten_effect_dict(effect)  # (d,)
            sig = DistConsistencyMemory._sig_from_ids(clean_ids, corrupt_ids)
            pen = EFFECT_MEM.penalty(logic, sig, e_now_flat)
            consistency_sum = pen if (consistency_sum is None) else (consistency_sum + pen)
            consistency_count += 1
            EFFECT_MEM.update_curr(logic, sig, e_now_flat)

            # clean caches
            purge_cache(clean_cache); purge_cache(corrupt_cache)
            clean_cache.cache.clear(); corrupt_cache.cache.clear()
            del clean_cache, corrupt_cache, out_clean, out_corrupt

    # Contrastive: pull(A,A') close; push(A,B) far
    flat = flatten_effects_to_embeddings(effects)
    keys = list(flat.keys())
    if len(keys) < 2 or any(len(flat[k]) < 2 for k in keys):
        # degenerate mini-batch: fallback zero loss
        return torch.zeros((), device=AP.DEVICE, dtype=DTYPE)

    if task_idx == 0:
        A = flat[keys[0]][0]; A_ = flat[keys[0]][1]; B = flat[keys[1]][0]
    else:
        A = flat[keys[1]][0]; A_ = flat[keys[1]][1]; B = flat[keys[0]][0]
        
    contrastive = nn.functional.cosine_similarity(A, A_, dim=0) / (nn.functional.cosine_similarity(A, A_, dim=0) + nn.functional.cosine_similarity(A, B, dim=0))
    contrastive = -torch.log(contrastive.clamp_min(1e-8))

    if consistency_count > 0 and args.consistency_lambda > 0.0:
        consistency = consistency_sum / float(consistency_count)
        loss = contrastive + (args.consistency_lambda * consistency)
    else:
        loss = contrastive

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loss


# ═══════════════════════ Reptile meta-training loop ═══════════════════════

print("[INFO] Start Reptile meta-training …")
for meta_iter in trange(args.meta_iters, desc="meta", colour="green"):
    # sample item; on StopIteration -> epoch end
    try:
        item = next(loader_iter)
        seen_in_epoch += 1
    except StopIteration:
        EFFECT_MEM.on_epoch_end()  # move curr → prev
        seen_in_epoch = 0
        loader_iter = iter(loader)
        item = next(loader_iter)
        seen_in_epoch += 1

    theta0 = clone_params(model)
    delta_sum = {k: torch.zeros_like(v).cpu() for k, v in theta0.items()}

    # one sampled item for all tasks in this meta-iter
    for task_idx in range(args.tasks_per_meta):
        with torch.no_grad():
            load_params_(model, theta0)
        inner_opt = SGD(model.parameters(), lr=args.inner_lr, momentum=0)
        for _ in range(args.inner_steps):
            loss = compute_task_loss(item, task_idx)
            inner_opt.zero_grad(set_to_none=True)
            loss.backward()
            inner_opt.step()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                delta_sum[k] += v.detach().cpu() - theta0[k].cpu()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        del inner_opt, loss

    # meta update
    for k, v in model.state_dict().items():
        v.data.add_(args.meta_lr * delta_sum[k].to(v.device) / max(1, args.tasks_per_meta))

    step = meta_iter + 1
    if (step % args.save_every == 0) or (step == args.meta_iters):
        ckpt_path = save_checkpoint(step, model, ckpt_dir=args.ckpt_dir, tag=args.save_tag)
        print(f"[INFO] saved checkpoint → {ckpt_path}")

    del theta0, delta_sum
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if (meta_iter + 1) % 50 == 0:
        print(f"[INFO] meta-iter {meta_iter + 1}: applied reptile update")

print("✓ Finished Reptile training.")