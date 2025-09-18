#!/usr/bin/env python3
"""
contrastive_plain_pgnull.py  —  No meta-learning, no Reptile.
Single-loop training with:
  • Contrastive effect loss (two “tasks” averaged)
  • Per-sample effect distribution consistency across epochs
  • Prediction-gradient null-space protection (anchor-based, optional)
  • Checkpointing
"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import trange

# ---------------------------------------------------------------------
# Project-local helpers
# ---------------------------------------------------------------------
from src.get_dataset import (
    LogicDataset,
    load_augmented_json_grouped,
    collate_fn,
)
import src.attribute_patch as AP  # exposes TOKENIZER / ActCacher / calculate_effect

# ---------------------------------------------------------------------
# CLI / minimal-argparse glue
# ---------------------------------------------------------------------
def _dataset_tag_from_path(p: Path) -> str:
    return Path(p).with_suffix("").name

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plain contrastive trainer (no meta-learning) + PG null projection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data/model
    ap.add_argument("--data_json", type=Path, default=Path("data/corrupt/level_1.json"))
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--group_size", type=int, default=2)
    ap.add_argument("--n_logic_per_item", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    # training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-6)

    # saving
    ap.add_argument("--ckpt_root", type=Path, default=Path("ckpts"))
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--save_tag", type=str, default="nometa")
    ap.add_argument("--dataset_tag", type=str, default=None,
                    help="Override dataset tag; default is stem of --data_json")

    # consistency (effect distribution across epochs)
    ap.add_argument("--consistency_lambda", type=float, default=0.1)
    ap.add_argument("--consistency_normed", type=int, choices=[0,1], default=1)

    # prediction-gradient null-space protection
    ap.add_argument("--prot_ratio", type=float, default=1.0,
                    help="0=no protection, 1=hard projection, (0,1)=soft")
    ap.add_argument("--project_every", type=int, default=1,
                    help="Apply projection every N steps")
    ap.add_argument("--eps_proj", type=float, default=1e-8)

    args = ap.parse_args()
    # derive dataset tag / ckpt dir
    tag = args.dataset_tag or _dataset_tag_from_path(args.data_json)
    args.dataset_tag = tag
    args.ckpt_dir = (args.ckpt_root / tag)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    return args

args = get_args()

# ---------------------------------------------------------------------
# Training hyper-parameters (wired from args)
# ---------------------------------------------------------------------
DATA_JSON: Path = args.data_json
GROUP_SIZE = args.group_size
N_LOGIC_PER_ITEM = args.n_logic_per_item
MAX_LEN = args.max_len
BATCH_SIZE = args.batch_size

EPOCHS = args.epochs
LR = args.lr
SEED = args.seed

CKPT_DIR = args.ckpt_dir
SAVE_EVERY = args.save_every
SAVE_TAG   = args.save_tag

# Distribution-constraint hyper-parameters
CONSISTENCY_LAMBDA = args.consistency_lambda
CONSISTENCY_NORMED = bool(args.consistency_normed)

DEVICE = AP.DEVICE
DTYPE = AP.DTYPE
MODEL_NAME = args.model_name or AP.MODEL_NAME

# ---------------------------------------------------------------------
# Distribution constraint memory
# ---------------------------------------------------------------------
class DistConsistencyMemory:
    def __init__(self, lam: float = 0.1, normed: bool = True):
        self.prev: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.curr: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.lam = float(lam)
        self.normed = bool(normed)

    @staticmethod
    def _sig_from_ids(clean_ids: torch.Tensor, corrupt_ids: torch.Tensor) -> str:
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

EFFECT_MEM = DistConsistencyMemory(lam=CONSISTENCY_LAMBDA, normed=CONSISTENCY_NORMED)

# ---------------------------------------------------------------------
# Utilities & checkpointing
# ---------------------------------------------------------------------
def save_checkpoint(step: int, model: nn.Module, tag: str = SAVE_TAG) -> Path:
    path = CKPT_DIR / f"{tag}_{step:05d}.pt"
    payload = {
        "iter": step,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "MODEL_NAME": MODEL_NAME,
            "GROUP_SIZE": GROUP_SIZE,
            "N_LOGIC_PER_ITEM": N_LOGIC_PER_ITEM,
            "MAX_LEN": MAX_LEN,
            "BATCH_SIZE": BATCH_SIZE,
            "SEED": SEED,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "DTYPE": str(DTYPE),
            "DATA_JSON": str(DATA_JSON),
            "DATASET_TAG": args.dataset_tag,
            "CONSISTENCY_LAMBDA": CONSISTENCY_LAMBDA,
            "CONSISTENCY_NORMED": CONSISTENCY_NORMED,
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
    torch.save(payload, CKPT_DIR / f"{tag}_latest.pt")
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

def purge_cache(cache: AP.ActCacher):
    for k, t in cache.cache.items():
        if t is not None:
            t.grad = None
        cache.cache[k] = None
    cache.cache.clear()

# ---------------------------------------------------------------------
# Build model / dataset / loader
# ---------------------------------------------------------------------
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

rows = load_augmented_json_grouped(DATA_JSON)
print(f"[INFO] Loaded {len(rows)} rows → grouping …")

ds = LogicDataset(
    data=rows,
    tokenizer=tok,
    group_size=GROUP_SIZE,
    n_logic_per_item=N_LOGIC_PER_ITEM,
    max_length=MAX_LEN,
    seed=SEED,
)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

nodes = AP.get_comp_nodes(model)
print(f"[INFO] Tracking {len(nodes)} computational nodes")

# ---------------------------------------------------------------------
# Prediction-gradient null-space protection helpers
# ---------------------------------------------------------------------
_proj_counter = 0

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
        s = torch.zeros((), device=p.device if 'p' in locals() else AP.DEVICE, dtype=DTYPE)
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

# ---------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------
def _forward_effects_for_item(item: Dict) -> Tuple[Dict[str, List[Dict[str, torch.Tensor]]],
                                                   Dict[str, torch.Tensor] | None]:
    """
    返回:
      effects: 逻辑模式 -> [effect_dict, ...]
      anchor_pg: 第一条样本若 calculate_effect 返回了 (effect, pred_grad_named)，则为该 dict；否则 None
    """
    effects: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    anchor_pg: Dict[str, torch.Tensor] | None = None
    took_anchor = False

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
                _ = model(**inputs_cor)

            ret = AP.calculate_effect(model, clean_cache, corrupt_cache, nodes, tok, out_clean, answers)
            if isinstance(ret, tuple) and len(ret) >= 2 and isinstance(ret[1], dict):
                effect, maybe_pg = ret[0], ret[1]
            else:
                effect, maybe_pg = ret, None

            effects[logic].append(effect)

            # 记录锚点样本的预测梯度（仅第一次）
            if (not took_anchor) and (maybe_pg is not None):
                anchor_pg = {k: v.detach() for k, v in maybe_pg.items()}
                took_anchor = True

            # effect consistency memory（按 token-id 签名存一份）
            e_now = flatten_effect_dict(effect)
            sig = DistConsistencyMemory._sig_from_ids(clean_ids, corrupt_ids)
            EFFECT_MEM.update_curr(logic, sig, e_now)

            # cleanup
            purge_cache(clean_cache); purge_cache(corrupt_cache)
            clean_cache.cache.clear(); corrupt_cache.cache.clear()
            del clean_cache, corrupt_cache, out_clean
    return effects, anchor_pg

def compute_combined_loss(item: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor] | None]:
    """
    Compute: 0.5 * contrastive(task=0) + 0.5 * contrastive(task=1) + consistency
    额外返回 anchor_pg（若 calculate_effect 提供）。
    """
    effects, anchor_pg = _forward_effects_for_item(item)

    flat = flatten_effects_to_embeddings(effects)
    keys = list(flat.keys())
    if len(keys) < 2 or len(flat[keys[0]]) < 2 or len(flat[keys[1]]) < 1:
        return torch.zeros([], device=AP.DEVICE, dtype=DTYPE), anchor_pg

    # Task 0
    A0 = flat[keys[0]][0]; A0_ = flat[keys[0]][1]; B0 = flat[keys[1]][0]
    loss0 = nn.functional.cosine_similarity(A0, A0_, dim=0) - nn.functional.cosine_similarity(A0, B0, dim=0)

    # Task 1
    A1 = flat[keys[1]][0]
    A1_ = flat[keys[1]][1] if len(flat[keys[1]]) > 1 else flat[keys[1]][0]
    B1 = flat[keys[0]][0]
    loss1 = nn.functional.cosine_similarity(A1, A1_, dim=0) - nn.functional.cosine_similarity(A1, B1, dim=0)

    # Consistency over the SAME samples vs previous epoch
    consistency_sum = None
    consistency_cnt = 0
    for logic, effect_list in effects.items():
        for eff in effect_list:
            e_now_flat = flatten_effect_dict(eff)
            sig = hashlib.blake2b(e_now_flat.detach().float().cpu().numpy().tobytes(), digest_size=12).hexdigest()
            # 当前 epoch 的缓存
            EFFECT_MEM.curr[logic][sig] = e_now_flat.detach().to("cpu", torch.float16).clone()
            # 与上个 epoch 对齐计算 penalty
            prev = EFFECT_MEM.prev.get(logic, {}).get(sig, None)
            if prev is not None:
                e_prev = prev.to(e_now_flat.device, e_now_flat.dtype)
                if CONSISTENCY_NORMED:
                    e1 = nn.functional.normalize(e_now_flat, dim=0, eps=1e-8)
                    e0 = nn.functional.normalize(e_prev,     dim=0, eps=1e-8)
                    pen = (1.0 - torch.sum(e1 * e0))
                else:
                    pen = nn.functional.mse_loss(e_now_flat, e_prev, reduction="mean")
                consistency_sum = pen if (consistency_sum is None) else (consistency_sum + pen)
                consistency_cnt += 1

    consistency_term = (consistency_sum / float(consistency_cnt)) if (consistency_cnt > 0 and CONSISTENCY_LAMBDA > 0) \
                       else torch.zeros([], device=AP.DEVICE, dtype=loss0.dtype)

    loss = 0.5 * (loss0 + loss1) + CONSISTENCY_LAMBDA * consistency_term
    return loss, anchor_pg

# ---------------------------------------------------------------------
# Train loop (single loop, no meta/reptile)
# ---------------------------------------------------------------------
print("[INFO] Start plain training …")
optimizer = SGD(model.parameters(), lr=LR, momentum=0.0)

global_step = 0
for epoch in range(EPOCHS):
    for item in loader:
        global_step += 1
        print(f"[INFO] Epoch {epoch+1}/{EPOCHS}, Step {global_step} ...")

        loss, anchor_pg = compute_combined_loss(item)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 关键：Prediction-gradient null-space protection（只基于当步 anchor_pg）
        try:
            apply_pg_null_projection_from_dict(
                anchor_pg,
                ratio=float(args.prot_ratio),
                every=int(args.project_every),
                eps=float(args.eps_proj),
            )
        except Exception as e:
            print(f"[WARN] pg-null projection skipped: {e}")

        optimizer.step()

        if (global_step % SAVE_EVERY) == 0:
            ckpt = save_checkpoint(global_step, model, tag=SAVE_TAG)
            print(f"[INFO] saved checkpoint @ step={global_step} → {ckpt}")

        # hygiene
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    EFFECT_MEM.on_epoch_end()
    print(f"[INFO] Finished epoch {epoch+1}/{EPOCHS}")

print("✓ Finished plain training.")