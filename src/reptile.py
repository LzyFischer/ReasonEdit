#!/usr/bin/env python3
"""contrastive_reptile.py

Train a causal-LM with **Reptile meta-learning** on a contrastive logic dataset.
Two contrastive losses are treated as two meta-tasks (Task-A / Task-B).

Requirements
------------
$ pip install torch higher tqdm

Key hyper-parameters are at the top of the file.
The script re-uses your original helper modules:
    • get_dataset.py
    • attribute_patch.py

Compared with the vanilla training loop, the changes are:
1. **Inner-loop**: run *k* SGD steps on each task starting from θ₀.
2. **Outer-loop**: parameter diff Δ = φ̃ − θ₀ is accumulated and applied
   by θ ← θ + ε·mean(Δ) (Reptile update, first-order ⇒ no Hessian).
3. **Memory hygiene**: ActCacher objects are purged after every forward.

If you want a first-order approximation of MAML (FOMAML) instead, set
`META_UPDATE="grad"` and keep `SECOND_ORDER=False` — but this script is
already first-order (no second-order gradients computed).
"""
from __future__ import annotations

import copy
import gc
from pathlib import Path
from typing import Dict, List
import pdb

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
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
GROUP_SIZE = 2          # see original script
N_LOGIC_PER_ITEM = 2
MAX_LEN = 256
BATCH_SIZE = 1

CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(exist_ok=True)
META_ITERS = 1000       # outer iterations
INNER_STEPS = 5         # k steps of SGD per task
INNER_LR = 1e-5
META_LR = 1e-4          # ε in Reptile
TASKS_PER_META = 2      # Task-A / Task-B
SEED = 0

DEVICE = AP.DEVICE
DTYPE = AP.DTYPE
MODEL_NAME = AP.MODEL_NAME

torch.manual_seed(SEED)

# -----------------------------------------------------------------------------
# Utility helpers: clone / load model parameters fast
# -----------------------------------------------------------------------------

def clone_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return *detached* deep copy of parameters."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_params_(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """In-place overwrite of model parameters without tracking grads."""
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

# -----------------------------------------------------------------------------
# Build model / dataset / loader (same as before)
# -----------------------------------------------------------------------------
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
# Task-specific loss builder (Task-A vs Task-B)
# -----------------------------------------------------------------------------

def compute_task_loss(item: Dict, task_idx: int) -> torch.Tensor:
    """Return single contrastive loss for given task (0 or 1)."""
    effects: Dict[str, List[Dict[str, torch.Tensor]]] = {}

    # ---------------- forward over group -----------------
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
                out_corrupt = model(**inputs_cor)

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



    # ---------------- contrastive computation -----------------
    flat = flatten_effects_to_embeddings(effects)
    if task_idx == 0:
        A = flat[list(flat.keys())[0]][0]
        A_ = flat[list(flat.keys())[0]][1]
        B = flat[list(flat.keys())[1]][0]
    else:
        A = flat[list(flat.keys())[1]][0]
        A_ = flat[list(flat.keys())[1]][1]
        B = flat[list(flat.keys())[0]][0]
    
    loss = nn.functional.cosine_similarity(A, A_, dim=0) - nn.functional.cosine_similarity(A, B, dim=0)

    purge_cache(clean_cache)
    purge_cache(corrupt_cache)
    clean_cache.cache.clear(), corrupt_cache.cache.clear()
    del clean_cache, corrupt_cache, out_clean, out_corrupt
    torch.cuda.empty_cache()

    return loss

# -----------------------------------------------------------------------------
# Reptile outer-loop training
# -----------------------------------------------------------------------------
print("[INFO] Start Reptile meta-training …")

for meta_iter in trange(META_ITERS, desc="meta", colour="green"):
    # theta0 = {k: v.detach().cpu() for k,v in model.state_dict().items()}
    theta0 = clone_params(model) 
    delta_sum = {k: torch.zeros_like(v).cpu() for k, v in theta0.items()}

    # -------- iterate over tasks per meta-iteration ----------
    item = next(loader_iter)
    for task_idx in range(TASKS_PER_META):
        # k inner steps
        with torch.no_grad():
            load_params_(model, theta0)
        inner_opt = SGD(model.parameters(), lr=INNER_LR, momentum=0)
        for _ in range(INNER_STEPS):
            loss = compute_task_loss(item, task_idx)
            inner_opt.zero_grad(set_to_none=True)
            loss.backward()
            inner_opt.step()

            # free non-leaf grads
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
        del inner_opt, loss

    # -------- meta update θ ← θ + ε · mean(Δ) -------------
    for k, v in model.state_dict().items():
        v.data.add_(META_LR * delta_sum[k].to(v.device) / TASKS_PER_META)
    
    ckpt_path = CKPT_DIR / f"reptile_{meta_iter+1:05d}.pt"
    torch.save(
        {
            "iter":   meta_iter + 1,
            "config": {                 # anything you might need to resume
                "META_ITERS": META_ITERS,
                "INNER_STEPS": INNER_STEPS,
                "META_LR": META_LR,
                "INNER_LR": INNER_LR,
                "MODEL_NAME": MODEL_NAME,
            },
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "rng_state":  torch.get_rng_state(),   # optional
        },
        ckpt_path,
    )
    print(f"[INFO] saved checkpoint → {ckpt_path}")
    
    del theta0, delta_sum
    # free memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    if (meta_iter + 1) % 50 == 0:
        print(f"[INFO] meta-iter {meta_iter+1}: applied reptile update")

print("✓ Finished Reptile training.")
