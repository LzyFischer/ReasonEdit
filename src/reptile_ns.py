#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contrastive_reptile_pgnull.py

Reptile meta-learning on a contrastive logic dataset, with **prediction-gradient
null-space protection**:

• Treat two contrastive losses as two meta-tasks (Task-A / Task-B).
• On each inner step, take the current batch's *anchor* prediction gradient g
  (∇θ log pθ(y|anchor)) and project the current parameter gradients onto the
  null space of g; strength controlled by --prot_ratio, frequency by --project_every.

All effect-space protection (Oja/PASTd + jvp/vjp) is removed.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

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
# Expose: DEVICE / DTYPE / MODEL_NAME / AutoTokenizer / AutoModelForCausalLM
#         ActCacher / calculate_effect
import src.attribute_patch as AP


# ═══════════════════════ Argparse / dataset tag ═══════════════════════

def dataset_tag_from_path(p: Path) -> str:
    return Path(p).with_suffix("").name

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Reptile meta-learning with prediction-gradient null-space protection",
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
    ap.add_argument("--save_tag", type=str, default="reptile_pgnull")
    ap.add_argument("--dataset_tag", type=str, default=None)

    # prediction-gradient null-space protection
    ap.add_argument("--prot_ratio", type=float, default=1.0,
                    help="0=no protection, 1=hard projection, (0,1)=soft")
    ap.add_argument("--project_every", type=int, default=1,
                    help="Apply projection every N inner steps")
    ap.add_argument("--eps_proj", type=float, default=1e-8)

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

# ═══════════════════════ Utility helpers ═══════════════════════

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
            "PROT_RATIO": args.prot_ratio,
            "PROJECT_EVERY": args.project_every,
            "EPS_PROJ": args.eps_proj,
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

# ═══════════════════════ PG-null projection helpers ═══════════════════════

_proj_counter = 0

@torch.no_grad()
def _dot_named(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> torch.Tensor:
    s = None
    dev = None
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        dev = p.device
        ga = a.get(name, None); gb = b.get(name, None)
        if ga is None or gb is None:
            continue
        val = (ga * gb).sum()
        s = val if s is None else s + val
    if s is None:
        s = torch.zeros((), device=dev or DEVICE, dtype=DTYPE)
    return s

def apply_pg_null_projection_from_dict(pg_named: Dict[str, torch.Tensor] | None,
                                       ratio: float,
                                       every: int = 1,
                                       eps: float = 1e-8):
    """
    grad <- grad - ratio * ((grad·g)/(g·g+eps)) * g
    其中 g 为锚点样本的“参数预测梯度”（按参数名对齐的字典）。
    """
    global _proj_counter
    if pg_named is None or ratio <= 0.0:
        return
    _proj_counter += 1
    if every > 1 and (_proj_counter % every) != 0:
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

# ═══════════════════════ Build model / dataset / loader ═══════════════════════

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

# ═══════════════════════ Task-specific loss (contrastive) ═══════════════════════

def compute_task_loss_and_anchor_pg(item: Dict, task_idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
    """
    Return:
      loss: contrastive loss for the given task (0 or 1)
      anchor_pg: dict[name->Tensor] if AP.calculate_effect provides it for the *first* sample; else None
    """
    effects: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    anchor_pg: Dict[str, torch.Tensor] | None = None
    took_anchor = False

    # Forward over group
    for logic, pair_list in item.items():
        effects[logic] = []
        for g1_dict in pair_list[0]:
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

            # 期望 AP.calculate_effect 返回 (effect_dict, pred_grad_named)
            ret = AP.calculate_effect(model, clean_cache, corrupt_cache, nodes, tok, out_clean, answers)
            if isinstance(ret, tuple) and len(ret) >= 2 and isinstance(ret[1], dict):
                effect, maybe_pg = ret[0], ret[1]
            else:
                effect, maybe_pg = ret, None

            effects[logic].append(effect)
            if (not took_anchor) and (maybe_pg is not None):
                anchor_pg = {k: v.detach() for k, v in maybe_pg.items()}
                took_anchor = True

            purge_cache(clean_cache); purge_cache(corrupt_cache)
            clean_cache.cache.clear(); corrupt_cache.cache.clear()
            del clean_cache, corrupt_cache, out_clean

    # Contrastive: pull(A,A') close; push(A,B) far
    flat = flatten_effects_to_embeddings(effects)
    keys = list(flat.keys())
    if len(keys) < 2 or any(len(flat[k]) < 2 for k in keys):
        return torch.zeros((), device=DEVICE, dtype=DTYPE), anchor_pg

    if task_idx == 0:
        A = flat[keys[0]][0]; A_ = flat[keys[0]][1]; B = flat[keys[1]][0]
    else:
        A = flat[keys[1]][0]; A_ = flat[keys[1]][1]; B = flat[keys[0]][0]

    loss = nn.functional.cosine_similarity(A, A_, dim=0) / (nn.functional.cosine_similarity(A, A_, dim=0) + nn.functional.cosine_similarity(A, B, dim=0))
    loss = -torch.log(loss.clamp_min(1e-8))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return loss, anchor_pg


# ═══════════════════════ Reptile outer-loop training ═══════════════════════

print("[INFO] Start Reptile meta-training …")

for meta_iter in trange(args.meta_iters, desc="meta", colour="green"):
    theta0 = clone_params(model)
    delta_sum = {k: torch.zeros_like(v).cpu() for k, v in theta0.items()}

    # sample one item for this meta-iter
    try:
        item = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        item = next(loader_iter)

    for task_idx in range(args.tasks_per_meta):
        with torch.no_grad():
            load_params_(model, theta0)

        inner_opt = SGD(model.parameters(), lr=args.inner_lr, momentum=0)

        for _ in range(args.inner_steps):
            loss, anchor_pg = compute_task_loss_and_anchor_pg(item, task_idx)
            inner_opt.zero_grad(set_to_none=True)
            loss.backward()

            # Prediction-gradient null-space projection (anchor-based)
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
    if (step % args.save_every) == 0 or (step == args.meta_iters):
        ckpt_path = save_checkpoint(step, model, ckpt_dir=args.ckpt_dir, tag=args.save_tag)
        print(f"[INFO] saved checkpoint → {ckpt_path}")

    del theta0, delta_sum
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if (meta_iter + 1) % 50 == 0:
        print(f"[INFO] meta-iter {meta_iter+1}: applied reptile update")

print("✓ Finished Reptile training.")