#!/usr/bin/env python3
"""
Run the head–level Edge-Attribution-Patch (EAP) in attribute_patch.py
on every (clean, corrupt) prompt pair that comes out of LogicDataset.

Assumes:
  • get_dataset (1).py is importable (or is in the same dir)
  • attribute_patch.py is importable and exposes
        – get_comp_nodes
        – token_logit_metric
        – ActivationCache
        – DEVICE, DTYPE, MODEL_NAME
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader

# --- your two files ----------------------------------------------------------
from src.get_dataset import (
    LogicDataset,
    load_augmented_json_grouped,         # helper in get_dataset (1).py
    chunk_indices,                       # if you need it elsewhere
    collate_fn
)
import src.attribute_patch as AP             # functions, CONFIG live here
import pdb
# -----------------------------------------------------------------------------
def purge_cache(cache):
    for k, t in cache.cache.items():
        t.grad = None          # ① 释放 grad
        cache.cache[k] = None  # ② 去掉对激活本身的引用
    cache.cache.clear()

def flatten_effect_dict(effect_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    # 将每个 tensor flatten，然后拼接起来
    flat_tensors = [v.flatten() for k, v in sorted(effect_dict.items())]
    return torch.cat(flat_tensors, dim=0)

def flatten_effects_to_embeddings(
    effects: Dict[str, List[Dict[str, torch.Tensor]]]
) -> Dict[str, List[torch.Tensor]]:
    flattened: Dict[str, List[torch.Tensor]] = {}

    for logic, effect_dicts in effects.items():
        if logic not in flattened:
            flattened[logic] = []
        for effect_dict in effect_dicts:
            # Sort keys to ensure consistent ordering
            flat_parts = [tensor.flatten() for key, tensor in sorted(effect_dict.items())]
            embedding = torch.cat(flat_parts, dim=0)
            flattened[logic].append(embedding)

    return flattened

def save_checkpoint(step: int, tag: str = "contrastive") -> Path:
    """Save model/optimizer and minimal config (CPU tensors for portability)."""
    fname = f"{tag}_{step:05d}.pt"
    path = CKPT_DIR / fname
    payload = {
        "step": step,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "MODEL_NAME": MODEL_NAME,
            "GROUP_SIZE": GROUP_SIZE,
            "N_LOGIC_PER_ITEM": N_LOGIC_PER_ITEM,
            "MAX_LEN": MAX_LEN,
            "BATCH_SIZE": BATCH_SIZE,
            "SEED": SEED,
            "LR": 1e-4,
        },
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "rng_state_cpu": torch.get_rng_state(),
    }
    # also store CUDA RNG if available
    if torch.cuda.is_available():
        try:
            payload["rng_state_cuda"] = torch.cuda.get_rng_state()
        except Exception:
            pass
    torch.save(payload, path)
    return path
# ------------------- 1.  hyper-params & paths  --------------------------------
DATA_JSON         = Path("data/corrupt/augmented_dataset.json")  # <-- adjust if needed
GROUP_SIZE        = 2
N_LOGIC_PER_ITEM  = 2
MAX_LEN           = 256
BATCH_SIZE        = 1
SEED              = 0
DEVICE            = AP.DEVICE
DTYPE             = AP.DTYPE
MODEL_NAME        = AP.MODEL_NAME
CKPT_DIR = Path("ckpts")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_EVERY = 10  # save every N steps; change as you like
# ------------------------------------------------------------------------------

# ------------------- 2.  model / tokenizer / dataset --------------------------
tok    = AP.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model  = AP.AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            attn_implementation="eager",
            device_map=None,
         ).to(DEVICE)
if hasattr(model.config, "sliding_window"):  # disable sliding-window for Qwen
    model.config.sliding_window = None
model.gradient_checkpointing_enable()

rows   = load_augmented_json_grouped(DATA_JSON)
ds     = LogicDataset(
            data=rows,
            tokenizer=tok,
            group_size=GROUP_SIZE,
            n_logic_per_item=N_LOGIC_PER_ITEM,
            max_length=MAX_LEN,
            seed=SEED,
         )
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
nodes  = AP.get_comp_nodes(model)
print(f"[INFO] Tracking {len(nodes)} comp-nodes across {len(ds)} dataset items")

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
# ------------------- 3.  per-batch attribution loop --------------------------
for step, item in enumerate(loader, 1):
    effects: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for i, (logic, pair_list) in enumerate(item.items()):
        effects[logic] = []
        for g1_dict in pair_list[0]:
            clean_ids    = g1_dict["clean_ids"].to(DEVICE)
            clean_mask   = g1_dict["clean_mask"].to(DEVICE)
            corrupt_ids  = g1_dict["corrupt_ids"].to(DEVICE)
            corrupt_mask = g1_dict["corrupt_mask"].to(DEVICE)
            answers      = g1_dict["answers_clean"]    # list[str]

            inputs_clean = {"input_ids": clean_ids, "attention_mask": clean_mask}
            inputs_cor   = {"input_ids": corrupt_ids, "attention_mask": corrupt_mask}

            # ------ 3-a  forward once to build clean & corrupt caches ----------
            
            clean_cache = AP.ActCacher(model, nodes)
            corrupt_cache = AP.ActCacher(model, nodes)
            with clean_cache:
                out_clean = model(**inputs_clean)
            with corrupt_cache:
                out_corrupt = model(**inputs_cor)
            
            effect = AP.calculate_effect(model, clean_cache, corrupt_cache, nodes, tok, out_clean, answers)
            effects[logic].append(effect)
            
            # AP.report_effects(effect, topk_node=5, topk_head=5)
    
    # contrastive learning loss (same logic similar, different not similar)
    # flatten effects for loss calculation into one embedding
    flattened_effects = flatten_effects_to_embeddings(effects)
    A = flattened_effects[list(flattened_effects.keys())[0]][0]  # first logic, first effect
    A_ = flattened_effects[list(flattened_effects.keys())[0]][1]  # first logic, second effect
    B = flattened_effects[list(flattened_effects.keys())[1]][0]  # second logic, first effect
    contrastive_loss = torch.nn.functional.cosine_similarity(A, A_, dim=0) - torch.nn.functional.cosine_similarity(A, B, dim=0)

    flattened_effects = flatten_effects_to_embeddings(effects)
    A = flattened_effects[list(flattened_effects.keys())[1]][0]  # first logic, first effect
    A_ = flattened_effects[list(flattened_effects.keys())[1]][1]  # first logic, second effect
    B = flattened_effects[list(flattened_effects.keys())[0]][0]  # second logic, first effect
    contrastive_loss += torch.nn.functional.cosine_similarity(A, A_, dim=0) - torch.nn.functional.cosine_similarity(A, B, dim=0)

    optimizer.zero_grad()
    contrastive_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (step % SAVE_EVERY) == 0:
        ck = save_checkpoint(step)
        print(f"[CKPT] saved → {ck}")

    print(f"[DONE] Step {step}/{len(loader)}")
    # clean all the cache and gradient
    purge_cache(clean_cache)
    purge_cache(corrupt_cache)
    clean_cache.cache.clear(), corrupt_cache.cache.clear()
    del clean_cache, corrupt_cache, out_clean, out_corrupt
    del effects
    del A, A_, B, contrastive_loss
    del flattened_effects
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # pdb.set_trace()  # uncomment to debug a specific batch


final_ck = save_checkpoint(step)
print(f"[CKPT] final → {final_ck}")
print("✓ All batches processed.")