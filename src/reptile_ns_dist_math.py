#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contrastive_reptile_pgnull.py

Reptile meta-learning + contrastive effect objective
with **anchor prediction-gradient null-space protection**.

• 每个 inner step：只用当下 batch 的 anchor sample，
  直接从 compute_task_loss 内部的 AP.calculate_effect 取出该样本的
  “参数预测梯度” g（第二返回值），然后把当前 .grad 投影到 g 的正交补。
• 保护强度用 --prot_ratio ∈ [0,1] 控制：0=不保护；1=硬投影；中间为软投影。

依赖：
$ pip install torch tqdm
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
import pdb

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import trange

# -----------------------------------------------------------------------------
# Project-local helpers
# -----------------------------------------------------------------------------
from src.get_dataset_math import (
    LogicDataset,
    load_augmented_json_grouped,
    collate_fn,
)
import src.attribute_patch as AP  # DEVICE, DTYPE, MODEL_NAME, ActCacher, calculate_effect

# =============================================================================
# Args
# =============================================================================

def dataset_tag_from_path(p: Path) -> str:
    p = Path(p)
    return p.stem

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Reptile meta-learning with prediction-gradient null-space protection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data / model
    ap.add_argument("--data_json", type=Path, default=Path("data/corrupt/math.json"))
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--group_size", type=int, default=2)
    ap.add_argument("--n_logic_per_item", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    # reptile / inner
    ap.add_argument("--meta_iters", type=int, default=45)
    ap.add_argument("--inner_steps", type=int, default=5)
    ap.add_argument("--inner_lr", type=float, default=1e-6)
    ap.add_argument("--meta_lr", type=float, default=1e-6)
    ap.add_argument("--tasks_per_meta", type=int, default=2)

    # saving
    ap.add_argument("--ckpt_root", type=Path, default=Path("ckpts"))
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--save_tag", type=str, default="reptile_nsmnl")
    ap.add_argument("--dataset_tag", type=str, default=None)

    # effect consistency (optional)
    ap.add_argument("--consistency_lambda", type=float, default=0.1)
    ap.add_argument("--consistency_normed", type=int, choices=[0, 1], default=1)

    # null-space protection
    ap.add_argument("--prot_ratio", type=float, default=1.0,
                    help="0=no protection, 1=hard projection, (0,1)=soft")
    ap.add_argument("--project_every", type=int, default=1,
                    help="Apply projection every N inner steps")
    ap.add_argument("--eps_proj", type=float, default=1e-8)

    args = ap.parse_args()
    tag = args.dataset_tag or dataset_tag_from_path(args.data_json)
    args.dataset_tag = tag
    args.ckpt_dir = (args.ckpt_root / tag)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    return args

args = get_args()

# =============================================================================
# Globals
# =============================================================================

DEVICE = AP.DEVICE
DTYPE = AP.DTYPE
MODEL_NAME = args.model_name or AP.MODEL_NAME

# =============================================================================
# Effect consistency (epoch-to-epoch)
# =============================================================================

class DistConsistencyMemory:
    def __init__(self, lam: float = 0.1, normed: bool = True):
        self.prev: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.curr: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.lam = float(lam)
        self.normed = bool(normed)

    @staticmethod
    def _sig_from_ids(clean_ids: torch.Tensor, corrupt_ids: torch.Tensor) -> str:
        import hashlib
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

# =============================================================================
# State utils
# =============================================================================

def clone_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def load_params_(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    for k, v in model.state_dict().items():
        v.data.copy_(state_dict[k].to(v.device, non_blocking=True))

def purge_cache(cache: AP.ActCacher):
    for k, t in cache.cache.items():
        if t is not None:
            t.grad = None
        cache.cache[k] = None
    cache.cache.clear()

def save_checkpoint(step: int, model: nn.Module, ckpt_dir: Path, tag: str) -> Path:
    path = ckpt_dir / f"{tag}_{step:05d}.pt"
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

# =============================================================================
# Effect helpers
# =============================================================================

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

# =============================================================================
# Build model / data
# =============================================================================

print("[INFO] Loading model …")
tok = AP.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AP.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    attn_implementation="eager",
    device_map=None,
).to(DEVICE)
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

nodes = AP.get_comp_nodes(model)
print(f"[INFO] Tracking {len(nodes)} computational nodes")

# =============================================================================
# Contrastive loss (effect-space) + consistency
# 返回 (loss, anchor_predgrad_named)
# =============================================================================

def compute_task_loss(item: Dict, task_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
    effects: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    consistency_sum = None
    consistency_count = 0
    anchor_pg: Dict[str, torch.Tensor] | None = None
    took_anchor = False

    for logic, pair_list in item.items():
        effects[logic] = []
        for idx, g1_dict in enumerate(pair_list[0]):
            clean_ids = g1_dict["clean_ids"].to(DEVICE)
            clean_mask = g1_dict["clean_mask"].to(DEVICE)
            corrupt_ids = g1_dict["corrupt_ids"].to(DEVICE)
            corrupt_mask = g1_dict["corrupt_mask"].to(DEVICE)
            answers = g1_dict["answers_clean"]

            inputs_clean = {"input_ids": clean_ids, "attention_mask": clean_mask}
            inputs_cor = {"input_ids": corrupt_ids, "attention_mask": corrupt_mask}

            clean_cache = AP.ActCacher(model, nodes)
            corrupt_cache = AP.ActCacher(model, nodes)

            with clean_cache:
                out_clean = model(**inputs_clean)
            with corrupt_cache:
                _ = model(**inputs_cor)

            # 兼容两种返回：1) effect  2) (effect, pred_grad_named)
            ret = AP.calculate_effect(
                model, clean_cache, corrupt_cache, nodes, tok, out_clean, answers,
            )
            if isinstance(ret, tuple) and len(ret) >= 2 and isinstance(ret[1], dict):
                effect, maybe_pg = ret[0], ret[1]
            else:
                effect, maybe_pg = ret, None

            effects[logic].append(effect)

            # 只用“第一个样本”作为 anchor（当步的 anchor）
            if (not took_anchor) and (maybe_pg is not None):
                anchor_pg = {k: v.detach() for k, v in maybe_pg.items()}
                took_anchor = True

            e_now_flat = flatten_effect_dict(effect)
            sig = DistConsistencyMemory._sig_from_ids(clean_ids, corrupt_ids)
            pen = EFFECT_MEM.penalty(logic, sig, e_now_flat)
            consistency_sum = pen if (consistency_sum is None) else (consistency_sum + pen)
            consistency_count += 1
            EFFECT_MEM.update_curr(logic, sig, e_now_flat)

            purge_cache(clean_cache); purge_cache(corrupt_cache)
            clean_cache.cache.clear(); corrupt_cache.cache.clear()
            del clean_cache, corrupt_cache, out_clean

    flat = flatten_effects_to_embeddings(effects)
    keys = list(flat.keys())
    if len(keys) < 2 or any(len(flat[k]) < 2 for k in keys):
        return torch.zeros((), device=DEVICE, dtype=DTYPE), anchor_pg

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
    return loss, anchor_pg

# =============================================================================
# Anchor pred-grad null projection (from dict provided by compute_task_loss)
# =============================================================================

_proj_step = 0

@torch.no_grad()
def _dot_named(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> torch.Tensor:
    s = None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ga = a.get(name, None); gb = b.get(name, None)
        if ga is None or gb is None:
            continue
        val = (ga * gb).sum()
        s = val if s is None else s + val
    if s is None:
        s = torch.zeros((), device=DEVICE, dtype=DTYPE)
    return s

def apply_pg_null_projection_from_dict(pg_named: Dict[str, torch.Tensor] | None,
                                       ratio: float,
                                       every: int = 1,
                                       eps: float = 1e-8):
    """
    grad <- grad - ratio * ((grad·g)/(g·g+eps)) * g
    其中 g 即 compute_task_loss 返回的 anchor_pg（参数名对齐的梯度字典）。
    """
    if pg_named is None or ratio <= 0.0:
        return
    global _proj_step
    _proj_step += 1
    if every > 1 and (_proj_step % every) != 0:
        return

    # 收集当前 .grad
    cur = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            cur[name] = torch.zeros_like(p) if (p.grad is None) else p.grad

    gg = _dot_named(pg_named, pg_named).clamp_min(eps)
    cg = _dot_named(cur, pg_named) / gg

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = pg_named.get(name, None)
        if g is None:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.add_( - ratio * cg * g )

# =============================================================================
# Train (Reptile)
# =============================================================================

print("[INFO] Start Reptile meta-training …")

for meta_iter in trange(args.meta_iters, desc="meta", colour="green"):
    try:
        item = next(loader_iter)
    except StopIteration:
        EFFECT_MEM.on_epoch_end()
        loader_iter = iter(loader)
        item = next(loader_iter)

    theta0 = clone_params(model)
    delta_sum = {k: torch.zeros_like(v).cpu() for k, v in theta0.items()}

    # one sampled item for all tasks
    for task_idx in range(args.tasks_per_meta):
        with torch.no_grad():
            load_params_(model, theta0)
        inner_opt = SGD(model.parameters(), lr=args.inner_lr, momentum=0)

        for _ in range(args.inner_steps):
            loss, anchor_pg = compute_task_loss(item, task_idx)
            inner_opt.zero_grad(set_to_none=True)
            loss.backward()

            # 直接使用 compute_task_loss 取到的 anchor_pg 进行投影
            try:
                apply_pg_null_projection_from_dict(
                    anchor_pg,
                    ratio=float(args.prot_ratio),
                    every=int(args.project_every),
                    eps=float(args.eps_proj),
                )
            except Exception as e:
                print(f"[WARN] pg-null projection skipped: {e}")

            inner_opt.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                delta_sum[k] += v.detach().cpu() - theta0[k].cpu()

        del inner_opt, loss, anchor_pg
        if torch.cuda.is_available():
            torch.cuda.ipc_collect(); torch.cuda.empty_cache()

    # meta update
    for k, v in model.state_dict().items():
        v.data.add_(args.meta_lr * delta_sum[k].to(v.device) / max(1, args.tasks_per_meta))

    step = meta_iter + 1
    if (step % args.save_every == 0) or (step == args.meta_iters):
        ckpt_path = save_checkpoint(step, model, ckpt_dir=args.ckpt_dir, tag=args.save_tag)
        print(f"[INFO] saved checkpoint → {ckpt_path}")

    del theta0, delta_sum
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()

print("✓ Finished Reptile training.")