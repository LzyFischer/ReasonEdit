#!/usr/bin/env python3
"""contrastive_reptile.py  (with null-space protected updates on lm_head)

Train a causal-LM with Reptile meta-learning on a contrastive logic dataset.
Adds null-space protection so that inner-step gradient updates on the output
head do not change logits for the *current batch* tokens.

Projection math:
  Given final hidden states H ∈ R^{N×d} (N tokens, d hidden),
  we seek ΔW such that ΔW @ h_i ≈ 0 ∀ i. The right-nullspace projector is
     P = I_d − Hᵀ (H Hᵀ + λI_N)^{-1} H
  and we update grad_W ← grad_W @ P.

Bias grad is zeroed to avoid logit shifts on current tokens.
"""
from __future__ import annotations

import copy
import gc
from pathlib import Path
from typing import Dict, List, Tuple
import pdb

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import trange
from datetime import datetime

# -----------------------------------------------------------------------------
# Project-local helpers (no change)
# -----------------------------------------------------------------------------
from src.get_dataset import (
    LogicDataset,
    load_augmented_json_grouped,
    collate_fn,
)
import src.attribute_patch as AP  # exposes TOKENIZER / ActCacher / calculate_effect

# -----------------------------------------------------------------------------
# Meta-learning hyper-parameters
# -----------------------------------------------------------------------------
DATA_JSON: Path = Path("data/corrupt/augmented_dataset.json")
GROUP_SIZE = 2
N_LOGIC_PER_ITEM = 2
MAX_LEN = 256
BATCH_SIZE = 1

CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(exist_ok=True)
META_ITERS = 1000
INNER_STEPS = 5
INNER_LR = 1e-5
META_LR = 1e-4
TASKS_PER_META = 2
SEED = 0

DEVICE = AP.DEVICE
DTYPE = AP.DTYPE
MODEL_NAME = AP.MODEL_NAME

torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# Utility helpers: clone / load model parameters fast
# -----------------------------------------------------------------------------
def clone_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def load_params_(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    for k, v in model.state_dict().items():
        v.data.copy_(state_dict[k].to(v.device, non_blocking=True))

# -----------------------------------------------------------------------------
# Cache housekeeping – re-use purge_cache from old script
# -----------------------------------------------------------------------------
def purge_cache(cache: AP.ActCacher):
    for k, t in cache.cache.items():
        t.grad = None
        cache.cache[k] = None
    cache.cache.clear()

# -----------------------------------------------------------------------------
# Effect flattening (unchanged helpers from previous script)
# -----------------------------------------------------------------------------
from typing import Dict as _Dict, List as _List
def flatten_effect_dict(effect_dict: _Dict[str, torch.Tensor]) -> torch.Tensor:
    flat = [v.flatten() for _, v in sorted(effect_dict.items())]
    return torch.cat(flat, 0)

def flatten_effects_to_embeddings(effects: _Dict[str, _List[_Dict[str, torch.Tensor]]]) -> _Dict[str, _List[torch.Tensor]]:
    flattened: _Dict[str, _List[torch.Tensor]] = {}
    for logic, effect_dicts in effects.items():
        flattened.setdefault(logic, [])
        for eff in effect_dicts:
            flat_parts = [t.flatten() for _, t in sorted(eff.items())]
            flattened[logic].append(torch.cat(flat_parts, 0))
    return flattened

# -----------------------------------------------------------------------------
# Build model / dataset / loader (same as before, but enable hidden states)
# -----------------------------------------------------------------------------
print("[INFO] Loading model …")

tok = AP.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AP.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    attn_implementation="eager",
    device_map=None,
).to(DEVICE)

# ensure we can fetch last hidden states
model.config.output_hidden_states = True
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
loader_iter = iter(loader)

nodes = AP.get_comp_nodes(model)
print(f"[INFO] Tracking {len(nodes)} computational nodes")

# -----------------------------------------------------------------------------
# Null-space projector utilities (NEW)
# -----------------------------------------------------------------------------
def get_output_head(model: nn.Module) -> nn.Linear:
    # HF causal LMs typically expose lm_head; fallback to get_output_embeddings()
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        return model.lm_head
    head = model.get_output_embeddings()
    if not isinstance(head, nn.Linear):
        raise RuntimeError("Unsupported output head type for null-space projection.")
    return head

@torch.no_grad()
def build_nullspace_projector(H: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """
    Stable projector via Woodbury:
      P = I - H^T (H H^T + λ I)^{-1} H  =  λ (H^T H + λ I)^{-1}
    Works entirely in d×d space (d = hidden size).

    H: [N, d] final hidden states (masked to actual tokens).
    Returns P: [d, d] (same device/dtype as H).
    """
    if H.numel() == 0:
        return torch.eye(1, device=H.device, dtype=H.dtype)[:0, :0]  # empty

    # Compute C = H^T H in float32 for stability
    H32 = H.detach().to(dtype=torch.float32)
    d = H32.shape[1]
    C = H32.transpose(0, 1) @ H32                          # [d, d]
    # A = C + λ I
    A = C + ridge * torch.eye(d, device=H32.device, dtype=H32.dtype)

    # Solve A X = I  (prefer solve/Cholesky over explicit inverse)
    I = torch.eye(d, device=H32.device, dtype=H32.dtype)
    try:
        X = torch.linalg.solve(A, I)                       # [d, d]
    except RuntimeError:
        # Fallback if solve complains (e.g., ill-conditioning): use cholesky
        L = torch.linalg.cholesky(A)
        X = torch.cholesky_inverse(L)

    P32 = ridge * X                                        # λ (C + λ I)^{-1}
    return P32.to(dtype=H.dtype)

def project_head_grad_onto_nullspace(model: nn.Module, H: torch.Tensor) -> None:
    """
    Right-multiply lm_head.weight.grad by projector P, and zero the bias grad.
    """
    head = get_output_head(model)
    if head.weight.grad is None:
        return
    P = build_nullspace_projector(H, ridge=1e-5)
    # grad_W: [V, d]; P: [d, d]  → project col-space
    head.weight.grad = head.weight.grad @ P
    if getattr(head, "bias", None) is not None and head.bias.grad is not None:
        # bias would uniformly shift logits; zero to preserve current predictions
        head.bias.grad.zero_()

# -----------------------------------------------------------------------------
# Task-specific loss builder (Task-A vs Task-B) – now returns hidden states too
# -----------------------------------------------------------------------------
def compute_task_loss_and_H(item: Dict, task_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, H) where H stacks final hidden states from forwards used."""
    effects: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    H_list: List[torch.Tensor] = []

    # ---------------- forward over group -----------------
    for logic, pair_list in item.items():
        effects[logic] = []
        for g1_dict in pair_list[0]:
            clean_ids = g1_dict["clean_ids"].to(DEVICE)
            clean_mask = g1_dict["clean_mask"].to(DEVICE)
            corrupt_ids = g1_dict["corrupt_ids"].to(DEVICE)
            corrupt_mask = g1_dict["corrupt_mask"].to(DEVICE)
            answers = g1_dict["answers_clean"]

            inputs_clean = {"input_ids": clean_ids, "attention_mask": clean_mask, "output_hidden_states": True}
            inputs_cor = {"input_ids": corrupt_ids, "attention_mask": corrupt_mask, "output_hidden_states": True}

            clean_cache = AP.ActCacher(model, nodes)
            corrupt_cache = AP.ActCacher(model, nodes)
            with clean_cache:
                out_clean = model(**inputs_clean)
            with corrupt_cache:
                out_corrupt = model(**inputs_cor)

            # collect last hidden states (exclude padding positions)
            last_clean = out_clean.hidden_states[-1]          # [B, T, d]
            last_corrupt = out_corrupt.hidden_states[-1]      # [B, T, d]
            # mask & flatten
            if clean_mask is not None:
                H_list.append(last_clean[clean_mask.bool()].reshape(-1, last_clean.size(-1)))
            else:
                H_list.append(last_clean.reshape(-1, last_clean.size(-1)))
            # if corrupt_mask is not None:
            #     H_list.append(last_corrupt[corrupt_mask.bool()].reshape(-1, last_corrupt.size(-1)))
            # else:
            #     H_list.append(last_corrupt.reshape(-1, last_corrupt.size(-1)))

            effect = AP.calculate_effect(
                model,
                clean_cache,
                corrupt_cache,
                nodes,
                tok,
                out_clean,  # logits already inside calculate_effect
                answers,
            )
            effects[logic].append(effect)

            # hygiene
            purge_cache(clean_cache)
            purge_cache(corrupt_cache)
            clean_cache.cache.clear(), corrupt_cache.cache.clear()
            del clean_cache, corrupt_cache

    # ---------------- contrastive computation -----------------
    flat = flatten_effects_to_embeddings(effects)
    if task_idx == 0:
        A  = flat[list(flat.keys())[0]][0]
        A_ = flat[list(flat.keys())[0]][1]
        B  = flat[list(flat.keys())[1]][0]
    else:
        A  = flat[list(flat.keys())[1]][0]
        A_ = flat[list(flat.keys())[1]][1]
        B  = flat[list(flat.keys())[0]][0]

    loss = nn.functional.cosine_similarity(A, A_, dim=0) - nn.functional.cosine_similarity(A, B, dim=0)

    # stack H (may be empty if something odd happened)
    H = torch.empty(0, model.config.hidden_size, device=DEVICE, dtype=DTYPE)
    if len(H_list) > 0:
        H = torch.cat(H_list, dim=0)

    # CUDA hygiene
    torch.cuda.empty_cache()
    return loss, H

# -----------------------------------------------------------------------------
# Reptile outer-loop training (with null-space projection)
# -----------------------------------------------------------------------------
print("[INFO] Start Reptile meta-training …")

for meta_iter in trange(META_ITERS, desc="meta", colour="green"):
    theta0 = clone_params(model)
    delta_sum = {k: torch.zeros_like(v).cpu() for k, v in theta0.items()}

    # fetch a fresh group
    item = next(loader_iter)

    # -------- iterate over tasks per meta-iteration ----------
    for task_idx in range(TASKS_PER_META):
        # reset to θ0
        with torch.no_grad():
            load_params_(model, theta0)

        inner_opt = SGD(model.parameters(), lr=INNER_LR, momentum=0)

        for _ in range(INNER_STEPS):
            loss, H = compute_task_loss_and_H(item, task_idx)
            inner_opt.zero_grad(set_to_none=True)
            loss.backward()

            # ---- NULL-SPACE PROJECTION on output head (NEW) ----
            if H.numel() > 0:
                project_head_grad_onto_nullspace(model, H)

            inner_opt.step()

            # free non-leaf grads
            del H, loss
            gc.collect()
            torch.cuda.empty_cache()

        # accumulate parameter delta
        with torch.no_grad():
            for k, v in model.state_dict().items():
                delta_sum[k] += v.detach().cpu() - theta0[k].cpu()

        # free memory
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        del inner_opt

    # -------- meta update θ ← θ + ε · mean(Δ) -------------
    for k, v in model.state_dict().items():
        v.data.add_(META_LR * delta_sum[k].to(v.device) / TASKS_PER_META)

    ckpt_path = CKPT_DIR / f"reptile_ns_{meta_iter+1:05d}.pt"
    torch.save(
        {
            "iter":   meta_iter + 1,
            "config": {
                "META_ITERS": META_ITERS,
                "INNER_STEPS": INNER_STEPS,
                "META_LR": META_LR,
                "INNER_LR": INNER_LR,
                "MODEL_NAME": MODEL_NAME,
            },
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "rng_state":  torch.get_rng_state(),
        },
        ckpt_path,
    )
    print(f"[INFO] saved checkpoint → {ckpt_path}")

    del theta0, delta_sum
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    if (meta_iter + 1) % 50 == 0:
        print(f"[INFO] meta-iter {meta_iter+1}: applied reptile update")

print("✓ Finished Reptile training.")