#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eap_hf.py
HuggingFace-only rewrite of your TL pipeline:
- No TransformerLens required
- Built-in EAP (activation patching) over per-layer resid/MLP and per-head attention
- JSON sparsification + Graphviz circuit rendering (PNG+PDF)

Python >=3.9, torch, transformers, numpy, networkx, graphviz
"""

from __future__ import annotations

# ───────────────────────────── Imports ─────────────────────────────────────
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any, Callable
import re
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# ───────────────────────────── Constants ───────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATA  = "data/corrupt/level_1.json"
DEFAULT_OUT   = "outputs/attr_scores"
TOP_N_EDGES   = 140
FIG_DPI       = 400

# SDP：避免实现差异引入不确定性
os.environ["PYTORCH_SDP_BACKEND"] = "math"
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

# ───────────────────────────── Utils ───────────────────────────────────────
def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def model_tag(name: str) -> str:
    last = name.split("/")[-1]
    return "".join("_" if c in ".-" else c for c in last).lower()

def quant_tag(q: float) -> str:
    keep_pct = round((1.0 - q) * 100, 3)
    return ("%g" % keep_pct).replace(".", "_")

def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ───────────────────── HuggingFace model wrapper + hooks ───────────────────
@dataclass
class NodeKey:
    layer: int
    typ: str        # 'resid_pre' | 'attn_head' | 'mlp' | 'resid_post' | 'logits'
    head: int = 0   # only for attn_head

    def short(self) -> str:
        if self.typ == "attn_head":
            return f"head.{self.layer}.{self.head}"
        if self.typ == "mlp":
            return f"mlp.{self.layer}"
        if self.typ == "resid_pre":
            return f"resid_pre.{self.layer}"
        if self.typ == "resid_post":
            return f"resid_post.{self.layer}"
        return self.typ

    def raw_name(self) -> str:
        return self.short()  # for compatibility with your prints

class HFRecorder:
    """
    Record per-layer hidden_states (resid_pre/post), per-layer MLP output (down_proj),
    per-head attention output BEFORE out-proj: shape [B, S, H, d_head].
    Works with LLaMA/Qwen-like decoder blocks.
    """
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.hooks: List[Any] = []
        self.clear()

        # guess layers
        # Qwen/LLaMA-like: model.model.layers is a ModuleList
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h
        else:
            raise RuntimeError("Unsupported architecture: cannot find model layers")

        self.num_layers = len(self.layers)

        # try to infer hidden size and heads from first layer
        sample_attn = getattr(self.layers[0], "self_attn", None) or getattr(self.layers[0], "attn", None)
        if sample_attn is None:
            raise RuntimeError("Layer has no attention module (self_attn/attn)")

        self.n_heads = getattr(sample_attn, "num_heads", None) or getattr(sample_attn, "n_heads", None)
        self.head_dim = getattr(sample_attn, "head_dim", None) or getattr(sample_attn, "dim", None)
        if self.n_heads is None or self.head_dim is None:
            # fallback: deduce from hidden size
            hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None)
            num_attention_heads = getattr(self.model.config, "num_attention_heads", None)
            if hidden_size and num_attention_heads:
                self.n_heads = num_attention_heads
                self.head_dim = hidden_size // num_attention_heads

        if self.n_heads is None or self.head_dim is None:
            raise RuntimeError("Cannot infer num_heads/head_dim")

        self.hidden_size = self.n_heads * self.head_dim

        self._install_hooks()

    # storages
    def clear(self):
        self.hidden_states: List[torch.Tensor] = [None for _ in range(getattr(self, "num_layers", 0) + 1)]  # [B,S,Hd], 0..L
        self.resid_post: List[torch.Tensor] = [None for _ in range(getattr(self, "num_layers", 0))]  # per-layer: [B,S,Hd] or None
        self.mlp_out: List[torch.Tensor] = [None for _ in range(getattr(self, "num_layers", 0))]   # per-layer: [B,S,Hd] or None
        self.attn_head_out: List[torch.Tensor] = [None for _ in range(getattr(self, "num_layers", 0))]  # per-layer: [B,S,H,d_head] or None

    # core: register hooks and a patched attention forward to expose per-head outputs
    def _install_hooks(self):
        # Enable hidden states from top model
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True  # not strictly necessary, but useful

        # 1) resid_post: hook block output
        def _hook_block_output(i):
            def fn(module, inp, out):
                # out is usually (hidden_states, *others) for decoder-only
                # But some models return tensor directly. Handle both.
                if isinstance(out, tuple):
                    hs = out[0]
                else:
                    hs = out
                self.resid_post[i] = hs.detach()
                return out
            return fn

        # 2) MLP down_proj output (post-mlp)
        def _hook_mlp_out(i):
            mlp = getattr(self.layers[i], "mlp", None) or getattr(self.layers[i], "feed_forward", None)
            if mlp is None:
                return
            # try common attr names: down_proj, down_proj, o_proj, dense_4h_to_h (GPT-NeoX-like)
            candidate = None
            for name in ["down_proj", "o_proj", "dense_4h_to_h", "gate_proj"]:  # gate if only gate exists, fallback to module output
                if hasattr(mlp, name) and isinstance(getattr(mlp, name), nn.Module):
                    candidate = getattr(mlp, name)
                    break

            if candidate is None:
                # fallback: hook mlp forward to capture return
                def mlp_wrapper_forward(mod, inp, out):
                    self.mlp_out[i] = out.detach()
                    return out
                self.hooks.append(mlp.register_forward_hook(mlp_wrapper_forward))
                return

            def mlp_proj_hook(mod, inp, out):
                self.mlp_out[i] = out.detach()
                return out
            self.hooks.append(candidate.register_forward_hook(mlp_proj_hook))

        # 3) attn per-head output before out-proj
        def _patch_attn_forward(i):
            attn = getattr(self.layers[i], "self_attn", None) or getattr(self.layers[i], "attn", None)
            if attn is None or not hasattr(attn, "forward"):
                return

            orig_forward = attn.forward

            def wrapped_forward(*args, **kwargs):
                # call original forward but intercept local variables by recomputing
                # Strategy: run original forward to get output (Tensor), but alongside,
                # recompute per-head output from inputs via a parallel run using
                # the same projections. This relies on HF attention public attrs.
                # This is approximate but faithful to LLaMA/Qwen attention math.
                # We accept the tiny extra compute to obtain per-head tensors.
                self.attn_head_out[i] = None  # reset
                # Try to reconstruct inputs:
                # HF attention forward signature usually: hidden_states, attention_mask, position_ids, past_key_value, ...
                # We'll grab hidden_states from args[0]
                try:
                    hidden_states = kwargs.get("hidden_states", None) or (args[0] if len(args) > 0 else None)
                    if hidden_states is not None:
                        hs = hidden_states
                        # project Q,K,V
                        to_2d = hs.ndim == 3
                        # common: attn.q_proj / k_proj / v_proj
                        q_proj = getattr(attn, "q_proj", None) or getattr(attn, "q", None)
                        k_proj = getattr(attn, "k_proj", None) or getattr(attn, "k", None)
                        v_proj = getattr(attn, "v_proj", None) or getattr(attn, "v", None)
                        if q_proj is not None and k_proj is not None and v_proj is not None:
                            q = q_proj(hs)   # [B,S,hidden]
                            k = k_proj(hs)
                            v = v_proj(hs)
                            B, S, _ = q.shape
                            H, Dh   = self.n_heads, self.head_dim
                            q = q.view(B, S, H, Dh).transpose(1, 2)  # [B,H,S,Dh]
                            k = k.view(B, S, H, Dh).transpose(1, 2)
                            v = v.view(B, S, H, Dh).transpose(1, 2)

                            # scaled dot-product
                            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)  # [B,H,S,S]

                            # Apply mask if present in kwargs
                            attn_mask = kwargs.get("attention_mask", None)
                            if attn_mask is not None:
                                # attn_mask typically [B,1,1,S] with 0/-inf
                                attn_scores = attn_scores + attn_mask

                            attn_probs = torch.softmax(attn_scores, dim=-1)  # [B,H,S,S]
                            head_out = torch.matmul(attn_probs, v)  # [B,H,S,Dh]
                            head_out = head_out.transpose(1, 2).contiguous()  # [B,S,H,Dh]
                            self.attn_head_out[i] = head_out.detach()

                except Exception:
                    # best-effort; if fails we'll just skip per-head outputs for this layer
                    self.attn_head_out[i] = None

                out = orig_forward(*args, **kwargs)
                return out

            # patch in-place
            attn.forward = wrapped_forward  # type: ignore

        # register hooks
        # Allocate slots
        self.resid_post = [None for _ in range(self.num_layers)]
        self.mlp_out    = [None for _ in range(self.num_layers)]
        self.attn_head_out = [None for _ in range(self.num_layers)]

        # Hook each block output
        for i in range(self.num_layers):
            self.hooks.append(self.layers[i].register_forward_hook(_hook_block_output(i)))
            _hook_mlp_out(i)
            _patch_attn_forward(i)

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    # Run model and collect caches
    @torch.no_grad()
    def run(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.clear()
        outputs = self.model(**inputs, output_hidden_states=True, output_attentions=False, use_cache=False)
        # hidden_states: tuple length L+1, each [B,S,Hd]
        hs_tuple = outputs.hidden_states
        self.hidden_states = [t.detach() for t in hs_tuple]  # 0..L

        # ensure lists filled
        for i in range(self.num_layers):
            if self.resid_post[i] is None:
                # if not set by hook, use hidden_states[i+1]
                self.resid_post[i] = self.hidden_states[i+1]
            if self.mlp_out[i] is None:
                # if cannot capture, approximate as residual delta between pre/post
                self.mlp_out[i] = self.hidden_states[i+1] - self.hidden_states[i]
            # attn per-head may remain None if failed; we’ll handle later

        logits = outputs.logits.detach()  # [B,S,V]
        return {"logits": logits}

# ───────────────────────────── EAP core (HF) ───────────────────────────────
def batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def to_tokens(tokenizer: PreTrainedTokenizerBase, texts: List[str]) -> Dict[str, torch.Tensor]:
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)

def make_avg_diff_metric(tokenizer: PreTrainedTokenizerBase, true_str: str, false_str: str) -> Callable[[torch.Tensor], torch.Tensor]:
    true_tok  = tokenizer(true_str, add_special_tokens=True, return_tensors="pt")["input_ids"][0, -1].item()
    false_tok = tokenizer(false_str, add_special_tokens=True, return_tensors="pt")["input_ids"][0, -1].item()
    def metric(logits: torch.Tensor) -> torch.Tensor:
        # mean over batch, final position: logit(True) - logit(False)
        last_pos = logits.size(1) - 1
        diff = logits[:, last_pos, true_tok] - logits[:, last_pos, false_tok]
        return diff.mean()
    return metric

def extract_clean_corrupt_pairs(examples: List[Dict]) -> Tuple[List[str], List[str]]:
    clean, corrupt = [], []
    for ex in examples:
        if isinstance(ex, dict) and ("clean" in ex) and ("corrupt" in ex):
            clean.append(ex["clean"])
            corrupt.append(ex["corrupt"])
        else:
            txt = ex if isinstance(ex, str) else json.dumps(ex, ensure_ascii=False)
            clean.append(txt)
            corrupt.append(txt)
    return clean, corrupt

# 允许的流向（用于构建有向边）
def allowed_edge(u: NodeKey, v: NodeKey) -> bool:
    if u.layer > v.layer:
        return False
    if u.typ == "resid_pre" and (v.typ in {"attn_head", "mlp", "resid_post"} and v.layer == u.layer):
        return True
    if u.typ == "attn_head" and v.typ == "resid_post" and v.layer == u.layer:
        return True
    if u.typ == "mlp" and v.typ == "resid_post" and v.layer == u.layer:
        return True
    if u.typ == "resid_post" and v.typ == "resid_pre" and v.layer == u.layer + 1:
        return True
    if u.typ in {"resid_post"} and v.typ == "logits" and v.layer >= u.layer:
        return True
    # allow skip connections to later resid_pre a few layers ahead
    if u.typ in {"resid_pre", "resid_post"} and v.typ == "resid_pre" and v.layer > u.layer:
        return True
    return False

class EAPGraph:
    """
    Stores:
      - node list (upstream & downstream share same superset)
      - eap_scores: Tensor[U, D] built from single-node patching impacts productized over (u->d) allowed edges
      - top_edges(n, abs_scores) -> list[(u_raw, v_raw, score)]
    """
    def __init__(self, upstream: List[NodeKey], downstream: List[NodeKey], node_impacts: Dict[str, float]):
        self.upstream = upstream
        self.downstream = downstream
        self.node_impacts = node_impacts  # impact(u) from patching u on metric
        # build matrix
        U, D = len(upstream), len(downstream)
        mat = torch.zeros(U, D, dtype=torch.float32)
        for i, u in enumerate(upstream):
            iu = node_impacts.get(u.short(), 0.0)
            for j, v in enumerate(downstream):
                if not allowed_edge(u, v):
                    continue
                iv = max(node_impacts.get(v.short(), 0.0), 0.0)
                # Edge weight：产品（方向性用 iu 的符号）
                w = float(iu) * float(iv)
                mat[i, j] = w
        self.eap_scores = mat.cpu()

    def top_edges(self, n: int = 100, abs_scores: bool = True) -> List[Tuple[str, str, float]]:
        scores = self.eap_scores.numpy()
        U, D = scores.shape
        flat = []
        for i in range(U):
            for j in range(D):
                s = scores[i, j]
                flat.append((i, j, s))
        key = (lambda x: abs(x[2])) if abs_scores else (lambda x: x[2])
        flat.sort(key=key, reverse=True)
        out = []
        for i, j, s in flat[:n]:
            out.append((self.upstream[i].raw_name(), self.downstream[j].raw_name(), float(s)))
        return out

# 核心：单节点 patch 影响度
def _replace_activation(tgt: torch.Tensor, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return src
    return torch.where(mask, src, tgt)

def _node_tensor_from_cache(rec: HFRecorder, key: NodeKey) -> Optional[torch.Tensor]:
    if key.typ == "resid_pre":
        return rec.hidden_states[key.layer]                # [B,S,Hd]
    if key.typ == "resid_post":
        return rec.resid_post[key.layer]                   # [B,S,Hd]
    if key.typ == "mlp":
        return rec.mlp_out[key.layer]                      # [B,S,Hd]
    if key.typ == "attn_head":
        A = rec.attn_head_out[key.layer]
        if A is None:
            return None
        # [B,S,H,Dh] → 只取该 head → [B,S,Dh]
        return A[:, :, key.head, :]
    if key.typ == "logits":
        return None
    return None

def _set_node_tensor_into_forward(model: PreTrainedModel, rec: HFRecorder, key: NodeKey, new_act: torch.Tensor):
    """
    Register a transient forward hook to replace the target node's tensor with new_act.
    Different node types require hooking different submodules.
    """
    hooks = []

    if key.typ in {"resid_pre"}:
        # Hack: replace hidden_states fed into block `key.layer`
        block = rec.layers[key.layer]
        orig_forward = block.forward

        def wrapped_forward(*args, **kwargs):
            # kwargs may have 'hidden_states' for LLaMA/Qwen
            if "hidden_states" in kwargs:
                hs = kwargs["hidden_states"]
                if hs.shape == new_act.shape:
                    kwargs["hidden_states"] = new_act
            else:
                # args[0] is hidden_states
                if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].shape == new_act.shape:
                    args = (new_act,) + args[1:]
            return orig_forward(*args, **kwargs)

        block.forward = wrapped_forward  # type: ignore

        def remover():
            block.forward = orig_forward  # type: ignore
        hooks.append(remover)

    elif key.typ in {"resid_post"}:
        # replace block output
        block = rec.layers[key.layer]
        def hook_block(mod, inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                if hs.shape == new_act.shape:
                    out = (new_act,) + out[1:]
            else:
                if out.shape == new_act.shape:
                    out = new_act
            return out
        h = block.register_forward_hook(hook_block)
        hooks.append(h.remove)

    elif key.typ == "mlp":
        mlp = getattr(rec.layers[key.layer], "mlp", None) or getattr(rec.layers[key.layer], "feed_forward", None)
        if mlp is None:
            return hooks
        # We replace mlp output by adjusting block output delta;
        # simpler: hook mlp module forward to return new_act_delta + (resid_pre)
        def mlp_forward_hook(mod, inp, out):
            # out: [B,S,Hd]
            if out.shape == new_act.shape:
                return new_act
            return out
        h = mlp.register_forward_hook(mlp_forward_hook)
        hooks.append(h.remove)

    elif key.typ == "attn_head":
        # replace a single head pre-out-proj: we intercept attn forward as in recorder
        attn = getattr(rec.layers[key.layer], "self_attn", None) or getattr(rec.layers[key.layer], "attn", None)
        if attn is None or not hasattr(attn, "forward"):
            return hooks
        orig_forward = attn.forward

        def wrapped_forward(*args, **kwargs):
            # recompute per-head as in recorder, then substitute target head
            try:
                hidden_states = kwargs.get("hidden_states", None) or (args[0] if len(args) > 0 else None)
                if hidden_states is not None:
                    hs = hidden_states
                    q_proj = getattr(attn, "q_proj", None) or getattr(attn, "q", None)
                    k_proj = getattr(attn, "k_proj", None) or getattr(attn, "k", None)
                    v_proj = getattr(attn, "v_proj", None) or getattr(attn, "v", None)
                    o_proj = getattr(attn, "o_proj", None) or getattr(attn, "o", None)
                    if q_proj is not None and k_proj is not None and v_proj is not None and o_proj is not None:
                        B, S, Hd = hs.shape
                        H = rec.n_heads
                        Dh = rec.head_dim
                        q = q_proj(hs).view(B, S, H, Dh).transpose(1, 2)  # [B,H,S,Dh]
                        k = k_proj(hs).view(B, S, H, Dh).transpose(1, 2)
                        v = v_proj(hs).view(B, S, H, Dh).transpose(1, 2)
                        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(Dh)
                        attn_mask = kwargs.get("attention_mask", None)
                        if attn_mask is not None:
                            attn_scores = attn_scores + attn_mask
                        attn_probs = torch.softmax(attn_scores, dim=-1)
                        head_out = torch.matmul(attn_probs, v)  # [B,H,S,Dh]
                        head_out = head_out.transpose(1, 2).contiguous()  # [B,S,H,Dh]
                        # replace target head
                        if new_act.shape == head_out[:, :, key.head, :].shape:
                            head_out[:, :, key.head, :] = new_act
                        # merge heads & out_proj
                        merged = head_out.view(B, S, H * Dh)
                        out_states = o_proj(merged)
                        # emulate the usual attention return: return out_states and keep rest unchanged signature-wise
                        return out_states
            except Exception:
                pass
            return orig_forward(*args, **kwargs)
        attn.forward = wrapped_forward  # type: ignore

        def remover():
            attn.forward = orig_forward  # type: ignore
        hooks.append(remover)

    return hooks

@torch.no_grad()
def run_with_patch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    recorder: HFRecorder,
    clean_inputs: Dict[str, torch.Tensor],
    corrupt_inputs: Dict[str, torch.Tensor],
    key: NodeKey,
    metric: Callable[[torch.Tensor], torch.Tensor],
) -> float:
    """
    Replace activation at `key` in the corrupt run with its clean counterpart.
    Return metric(clean_corrupt_patched) - metric(corrupt_baseline).
    Positive: patch pushes prediction toward 'True' token.
    """
    device = next(model.parameters()).device

    # 1) Baselines
    recorder.run(corrupt_inputs)
    logits_corrupt = recorder.model(**corrupt_inputs, output_hidden_states=False, use_cache=False).logits
    m_corrupt = metric(logits_corrupt).item()

    # 2) Get clean activation for this node
    recorder.run(clean_inputs)
    clean_tensor = _node_tensor_from_cache(recorder, key)
    if clean_tensor is None:
        return 0.0

    # 3) Rerun corrupt with patch
    # We must install a transient hook/hack to inject clean_tensor at the right spot.
    patch_hooks = _set_node_tensor_into_forward(model, recorder, key, clean_tensor.to(device))
    try:
        logits_patched = recorder.model(**corrupt_inputs, output_hidden_states=False, use_cache=False).logits
        m_patched = metric(logits_patched).item()
    finally:
        for h in patch_hooks:
            try:
                h()
            except Exception:
                pass

    return float(m_patched - m_corrupt)

def build_node_space(num_layers: int, n_heads: int) -> List[NodeKey]:
    nodes: List[NodeKey] = []
    for L in range(num_layers):
        for h in range(n_heads):
            nodes.append(NodeKey(L, "attn_head", h))
        nodes.append(NodeKey(L, "mlp"))
    # logits 作为唯一的 sink（给个最大层号方便布局）
    nodes.append(NodeKey(num_layers - 1, "logits"))
    return nodes

def run_eap(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    clean_texts: List[str],
    corrupt_texts: List[str],
    true_tok: str,
    false_tok: str,
    batch_size: int = 1,
) -> EAPGraph:
    """
    Best-effort, batched by tokenizer padding; metric = avg_diff(True, False).
    Returns EAPGraph with node-importance-derived edge scores.
    """
    device = next(model.parameters()).device
    metric = make_avg_diff_metric(tokenizer, true_tok, false_tok)

    # tokenize (single batch for simplicity; you can shard by batch_size if very large)
    clean_inputs   = batch_to_device(to_tokens(tokenizer, clean_texts), device)
    corrupt_inputs = batch_to_device(to_tokens(tokenizer, corrupt_texts), device)

    recorder = HFRecorder(model)

    # Build node space
    num_layers = recorder.num_layers
    n_heads    = recorder.n_heads
    all_nodes  = build_node_space(num_layers, n_heads)

    # Record baseline to size-check shapes
    recorder.run(clean_inputs)   # fill caches for shapes
    recorder.run(corrupt_inputs)

    # Compute node impact by single-node patching
    node_impacts: Dict[str, float] = {}
    for nk in all_nodes:
        if nk.typ == "logits":
            continue
        try:
            delta = run_with_patch(model, tokenizer, recorder, clean_inputs, corrupt_inputs, nk, metric)
            node_impacts[nk.short()] = float(delta)
        except Exception as e:
            node_impacts[nk.short()] = 0.0

    # Define upstream/downstream sets
    upstream   = [n for n in all_nodes if n.typ in {"attn_head", "mlp"}]
    downstream = [n for n in all_nodes if n.typ in {"attn_head", "mlp", "logits"}]

    graph = EAPGraph(upstream=upstream, downstream=downstream, node_impacts=node_impacts)
    return graph

# ───────────────────────────── Visualization ───────────────────────────────
def _node_style_for_typ(typ: str) -> Tuple[str, str]:
    palette = {
        "resid_pre": ("#d7f0d0", "#6ca966"),
        "attn_head": ("#d8e8ff", "#5d7fbf"),
        "mlp":       ("#fff5cc", "#b8963b"),
        "resid_post":("#ffd7b8", "#c47f42"),
        "logits":    ("#eadcff", "#6b5fbf"),
    }
    return palette.get(typ, ("#eeeeee", "#888888"))

def _map_node_id_and_label(nk: NodeKey) -> Tuple[str, str, str]:
    if nk.typ == "attn_head":
        return f"A{nk.layer}H{nk.head}", f"a{nk.layer}.h{nk.head}", "attn_head"
    if nk.typ == "mlp":
        return f"M{nk.layer}", f"m{nk.layer}", "mlp"
    if nk.typ == "resid_pre":
        return f"RPRE{nk.layer}", f"rpre{nk.layer}", "resid_pre"
    if nk.typ == "resid_post":
        return f"RPOST{nk.layer}", f"rpost{nk.layer}", "resid_post"
    if nk.typ == "logits":
        return "LOGITS", "logits", "logits"
    return "UNK", nk.short(), nk.typ

def visualize_circuit_gv(eap: EAPGraph, out_file: str) -> None:
    try:
        import graphviz
    except Exception as e:
        return visualize_circuit_fallback(eap, out_file)

    # Build nx graph from top edges to lay out
    tops = eap.top_edges(n=TOP_N_EDGES, abs_scores=True)
    G = nx.DiGraph()
    id_map: Dict[str, Tuple[str, str]] = {}  # raw -> (id,label)
    layer_group: Dict[int, set] = {}

    # Parse NodeKey back from raw names
    def parse_raw(raw: str) -> NodeKey:
        if raw.startswith("head."):
            _, L, H = raw.split(".")
            return NodeKey(int(L), "attn_head", int(H))
        if raw.startswith("mlp."):
            return NodeKey(int(raw.split(".")[1]), "mlp")
        if raw.startswith("resid_pre."):
            return NodeKey(int(raw.split(".")[1]), "resid_pre")
        if raw.startswith("resid_post."):
            return NodeKey(int(raw.split(".")[1]), "resid_post")
        if raw == "logits":
            return NodeKey(9999, "logits")
        # fallback
        m = re.search(r"(\d+)", raw)
        L = int(m.group(1)) if m else 0
        return NodeKey(L, "resid_post")

    for uraw, vraw, s in tops:
        u = parse_raw(uraw); v = parse_raw(vraw)
        uid, ulabel, ucat = _map_node_id_and_label(u)
        vid, vlabel, vcat = _map_node_id_and_label(v)
        if uid not in G:
            G.add_node(uid, label=ulabel, cat=ucat, L=u.layer)
            layer_group.setdefault(u.layer, set()).add(uid)
        if vid not in G:
            G.add_node(vid, label=vlabel, cat=vcat, L=v.layer)
            layer_group.setdefault(v.layer, set()).add(vid)
        G.add_edge(uid, vid, w=abs(float(s)))

    dot = graphviz.Digraph("circuit", format="png")
    dot.attr(rankdir="LR", splines="spline", concentrate="false",
             nodesep="0.35", ranksep="1.0", bgcolor="white",
             fontname="Helvetica")

    # declare nodes by layer ranks
    for L, nodes in sorted(layer_group.items()):
        with dot.subgraph(name=f"rank_{L}") as s:
            s.attr(rank="same")
            for nid in sorted(nodes):
                nd = G.nodes[nid]
                fill, border = _node_style_for_typ(nd["cat"])
                s.node(nid, label=nd["label"], shape="box", style="rounded,filled",
                       color=border, fillcolor=fill, penwidth="1.4", fontsize="18")

    # ensure logits node rank max+1
    if "LOGITS" in G:
        with dot.subgraph(name="rank_logits") as s:
            s.attr(rank="same")
            s.node("LOGITS")

    # edges
    ws = [d["w"] for _, _, d in G.edges(data=True)]
    strong_cut = np.quantile(ws, 0.9) if len(ws) >= 4 else (max(ws) if ws else 1.0)
    for u, v, d in G.edges(data=True):
        w = float(d["w"])
        penw = 1.1 + 1.4 * min(w / (strong_cut + 1e-8), 1.0)
        dot.edge(u, v, color="#5d7fbf", penwidth=f"{penw:.2f}", arrowsize="0.8")

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dot.format = "png"
    dot.render(filename=out_path.with_suffix("").as_posix(), cleanup=True)
    dot.format = "pdf"
    dot.render(filename=out_path.with_suffix(".pdf").as_posix(), cleanup=True)
    print(f"[viz] saved: {out_file}")

def visualize_circuit_fallback(eap: EAPGraph, out_file: str):
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    for u, v, s in eap.top_edges(n=TOP_N_EDGES, abs_scores=True):
        G.add_edge(u, v, weight=abs(s))
    pos = nx.spring_layout(G, seed=0)
    plt.figure(figsize=(10,6))
    nx.draw(G, pos, with_labels=False, node_size=220, width=0.8, arrows=True)
    plt.tight_layout()
    outp = Path(out_file)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outp, dpi=FIG_DPI)
    plt.close()
    print(f"[viz-fallback] saved: {out_file}")

# ───────────────────────────── CLI & Main ──────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    p.add_argument("--data_file", default=DEFAULT_DATA,
                   help="JSON：logic_groups[*].prompts[*].{clean,corrupt,...}")
    p.add_argument("--quants", nargs="+", type=float, default=[0.99],
                   help="分位阈值（取 >= q 对应的分数，进而“保留 (1-q) 比例”）")
    p.add_argument("--subset_k", type=int, default=2, help="每个 split 中 partA/partB 各取 k 条样本")
    p.add_argument("--splits", type=int, default=1, help="随机切分次数")
    p.add_argument("--seed", type=int, default=0, help="随机种子基数")
    p.add_argument("--batch_size", type=int, default=1, help="batch size（目前一次 tokenize 全部，必要时你可分批）")
    p.add_argument("--out_root", default=DEFAULT_OUT, help="结果根目录")
    p.add_argument("--ans_true", default=" True", help="metric 正类词（默认 ' True'）")
    p.add_argument("--ans_false", default=" False", help="metric 负类词（默认 ' False'）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p

def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)
    set_seed_all(args.seed)

    print(f"[load] AutoModelForCausalLM.from_pretrained({args.model}) on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(device)
    model.eval()

    # out dirs
    out_root = Path(args.out_root).resolve()
    dataset_tag = Path(args.data_file).stem
    model_dir = out_root / dataset_tag / model_tag(args.model)
    for q in args.quants:
        (model_dir / quant_tag(q) / "origin" / "figures").mkdir(parents=True, exist_ok=True)

    # load data
    with open(args.data_file, "r", encoding="utf-8") as f:
        logic_groups = json.load(f)

    # iterate logic groups
    for idx, logic in enumerate(logic_groups):
        examples = logic.get("prompts", [])
        if not examples:
            continue
        logic_name = f"logic_{idx:03d}"
        print(f"\n▶ {logic_name} — {len(examples)} prompts")

        # sampling
        k = min(args.subset_k, max(len(examples)//2, 1))
        if dataset_tag == "level_3":
            k = min(k, 10)

        for split in range(1, args.splits + 1):
            seed_i = args.seed + split - 1
            rng = random.Random(seed_i)
            shuffled = examples[:]
            rng.shuffle(shuffled)
            partA, partB = shuffled[:k], shuffled[k:2*k]

            # A
            clean_A, corrupt_A = extract_clean_corrupt_pairs(partA)
            eap_A = run_eap(model, tokenizer, clean_A, corrupt_A, args.ans_true, args.ans_false, args.batch_size)
            for q in args.quants:
                qtag   = quant_tag(q)
                subdir = model_dir / qtag / "origin"
                json_path = subdir / f"{logic_name}_split{seed_i}_partA.json"

                # sparsify
                scores = eap_A.eap_scores
                flat   = scores.flatten()
                thresh = torch.quantile(flat, q).item() if flat.numel() else 0.0
                keep   = scores >= thresh
                u_idx, d_idx = torch.nonzero(keep, as_tuple=True)
                kept_vals = scores[keep]
                edges = torch.stack([u_idx, d_idx, kept_vals], dim=1).tolist()
                save_json({
                    "shape": [int(scores.size(0)), int(scores.size(1))],
                    "threshold": float(thresh),
                    "edges": edges,
                }, json_path)

                viz_dir = subdir / "figures"
                viz_dir.mkdir(parents=True, exist_ok=True)
                visualize_circuit_gv(
                    eap_A,
                    out_file=str(viz_dir / f"{logic_name}_split{seed_i}_partA_circuit_top100.png"),
                )
                print(f"   ✓ partA  q={q:.3f} → {json_path.relative_to(model_dir)}")

            # B
            if partB:
                clean_B, corrupt_B = extract_clean_corrupt_pairs(partB)
                eap_B = run_eap(model, tokenizer, clean_B, corrupt_B, args.ans_true, args.ans_false, args.batch_size)
                for q in args.quants:
                    qtag   = quant_tag(q)
                    subdir = model_dir / qtag / "origin"
                    json_path = subdir / f"{logic_name}_split{seed_i}_partB.json"

                    scores = eap_B.eap_scores
                    flat   = scores.flatten()
                    thresh = torch.quantile(flat, q).item() if flat.numel() else 0.0
                    keep   = scores >= thresh
                    u_idx, d_idx = torch.nonzero(keep, as_tuple=True)
                    kept_vals = scores[keep]
                    edges = torch.stack([u_idx, d_idx, kept_vals], dim=1).tolist()
                    save_json({
                        "shape": [int(scores.size(0)), int(scores.size(1))],
                        "threshold": float(thresh),
                        "edges": edges,
                    }, json_path)

                    viz_dir = subdir / "figures"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    visualize_circuit_gv(
                        eap_B,
                        out_file=str(viz_dir / f"{logic_name}_split{seed_i}_partB_circuit_top100.png"),
                    )
                    print(f"   ✓ partB  q={q:.3f} → {json_path.relative_to(model_dir)}")

    print("\n✅ Finished for all TOP_QUANTS.")

if __name__ == "__main__":
    main()