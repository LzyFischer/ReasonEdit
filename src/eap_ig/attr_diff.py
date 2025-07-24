# attr_diff.py
from __future__ import annotations
from typing import Callable, List, Optional, Literal, Tuple, Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum

from .graph import Graph, AttentionNode, LogitNode

# --------------------------- helpers ---------------------------

@torch.no_grad()
def _check_cfg(model: HookedTransformer): ### This is for ensuring the model is compatible for circuit
    assert model.cfg.use_attn_result, "need cfg.use_attn_result=True"
    assert model.cfg.use_split_qkv_input, "need cfg.use_split_qkv_input=True"
    assert model.cfg.use_hook_mlp_in, "need cfg.use_hook_mlp_in=True"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "need cfg.ungroup_grouped_query_attention=True"

def _tok_plus(model: HookedTransformer, texts: List[str]):
    from .utils import tokenize_plus   # reuse
    return tokenize_plus(model, texts)

# ----------------- capture activations & grads -----------------
class CaptureStore:
    def __init__(self, graph, B, n_pos, d_model, device, dtype):
        self.graph = graph
        self.delta_src = torch.zeros(B, n_pos, graph.n_forward, d_model, device=device, dtype=dtype)
        self.scores    = torch.zeros(graph.n_forward, graph.n_backward, device=device, dtype=dtype)
        self.device    = device
        self.dtype     = dtype

    def add_src_act(self, fwd_idx, acts, sign):
        if acts.ndim == 4:  # [B,P,H,d_head]
            acts = acts.flatten(-2, -1)
        self.delta_src[:, :, fwd_idx] += sign * acts

    def accumulate(self, prev_i, bwd_i, grad_tensor, chunk=128):
        # grad_tensor: [B,P,d_model] 或 [B,P,H,d_head]
        g = grad_tensor
        if g.ndim > 3:
            g = g.flatten(-2, -1)
        f_slice = self.delta_src[:, :, :prev_i]           # (B,P,F,Hs)
        Hs_src = f_slice.size(-1); Hs_dst = g.size(-1)
        if Hs_src != Hs_dst:
            if Hs_dst % Hs_src == 0:
                g = g.view(g.size(0), g.size(1), Hs_src, -1).mean(-1)
            elif Hs_src % Hs_dst == 0:
                f_slice = f_slice.view(f_slice.size(0), f_slice.size(1), f_slice.size(2), Hs_dst, -1).mean(-1)
                Hs_src = Hs_dst
            else:
                return  # 跳过不配对的情况或 raise

        for s in range(0, prev_i, chunk):
            e = min(prev_i, s + chunk)
            contrib_chunk = (f_slice[:, :, s:e] * g.unsqueeze(2)).sum(dim=(0,1,3))  # (chunk,)
            if isinstance(bwd_i, slice):
                self.scores[s:e, bwd_i] = self.scores[s:e, bwd_i] + contrib_chunk.unsqueeze(1)
            else:
                self.scores[s:e, bwd_i] = self.scores[s:e, bwd_i] + contrib_chunk


def _make_capture_hooks(model: HookedTransformer, graph: Graph, store: CaptureStore):
    """Return 3 hook lists:
       corrupted_hooks: add corrupted activations (+1)
       clean_hooks    : add clean activations (-1)
       dst_hooks      : wrap dst-input tensors (requires_grad) & record mapping
    """
    corrupted_hooks, clean_hooks, dst_hooks = [], [], []

    # ---------- helpers ----------
    def src_hook_scalar(fwd_idx: int, sign: int):
        def hook_fn(acts, hook):
            store.add_src_act(fwd_idx, acts, sign)
            return acts
        return hook_fn

    def src_hook_attn(layer: int, sign: int):
        base = graph.forward_index(graph.nodes[f"a{layer}.h0"], attn_slice=False)
        def hook_fn(acts, hook):
            # acts: [B, P, n_heads, d_head]
            B, P, H, Hd = acts.shape
            for h in range(H):
                store.add_src_act(base + h, acts[..., h, :], sign)
            return acts
        return hook_fn

    def dst_in_hook(prev_i, bwd_i):
        def hook_fn(acts, hook):
            acts = acts.requires_grad_(True)
            def grad_cb(g):
                # g: grad wrt this dst-input
                store.accumulate(prev_i, bwd_i, g)
            acts.register_hook(grad_cb)
            return acts
        return hook_fn
    # ---------- input node ----------
    node_in = graph.nodes["input"]
    f_in = graph.forward_index(node_in, attn_slice=False)
    corrupted_hooks.append((node_in.out_hook, src_hook_scalar(f_in, +1)))
    clean_hooks.append((node_in.out_hook, src_hook_scalar(f_in, -1)))

    # ---------- per layer ----------
    for layer in range(graph.cfg["n_layers"]):
        # attention outputs (one hook handles all heads)
        attn0 = graph.nodes[f"a{layer}.h0"]
        corrupted_hooks.append((attn0.out_hook, src_hook_attn(layer, +1)))
        clean_hooks.append((attn0.out_hook, src_hook_attn(layer, -1)))

        # attention qkv inputs (dst of edges)
        prev_i = graph.prev_index(attn0)
        for i, letter in enumerate("qkv"):
            bwd_i = graph.backward_index(attn0, qkv=letter, attn_slice=False)
            dst_hooks.append((attn0.qkv_inputs[i], dst_in_hook(prev_i, bwd_i)))

        # mlp out
        mlp = graph.nodes[f"m{layer}"]
        f_m = graph.forward_index(mlp, attn_slice=False)
        corrupted_hooks.append((mlp.out_hook, src_hook_scalar(f_m, +1)))
        clean_hooks.append((mlp.out_hook, src_hook_scalar(f_m, -1)))

        # mlp in (dst)
        prev_i = graph.prev_index(mlp)
        bwd_i = graph.backward_index(mlp, attn_slice=False)
        dst_hooks.append((mlp.in_hook, dst_in_hook(prev_i, bwd_i)))

    # ---------- logits dst ----------
    log_node = graph.nodes["logits"]
    prev_i = graph.prev_index(log_node)
    bwd_i = graph.backward_index(log_node, attn_slice=False)
    dst_hooks.append((log_node.in_hook, dst_in_hook(prev_i, bwd_i)))

    return corrupted_hooks, clean_hooks, dst_hooks

# ------------------- main differentiable scorer -------------------

@torch.enable_grad()
def eap_ig_inputs_diff(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor, Tensor, Tensor, List[str]], Tensor],
    steps: int = 30,
    quiet: bool = False,
) -> Tensor:
    """
    Differentiable EAP-IG on *inputs*.
    Returns scores: Tensor[F, B] with requires_grad=True.
    No .backward() inside; uses autograd.grad to keep graph.
    """
    _check_cfg(model)
    device = model.cfg.device
    dtype  = model.cfg.dtype
    nF, nB = graph.n_forward, graph.n_backward

    score_acc = None
    n_items   = 0
    total_steps = 0

    data_iter = dataloader if quiet else tqdm(dataloader)

    for clean, corrupt, label in data_iter:
        B = len(clean)
        clean_toks, attn_mask, lengths, n_pos = _tok_plus(model, clean)
        corrupt_toks, _, _, n_pos_c = _tok_plus(model, corrupt)
        assert n_pos == n_pos_c, "clean/corrupt length mismatch"

        store = CaptureStore(graph, B, n_pos, model.cfg.d_model, model.cfg.device, model.cfg.dtype)
        ch_cor, ch_clean, dst_hooks = _make_capture_hooks(model, graph, store)

        # ---------- forward corrupted once ----------
        with model.hooks(fwd_hooks=ch_cor):
            _ = model(corrupt_toks, attention_mask=attn_mask)

        # Keep a copy of corrupted input activations
        corrupted_input = store.delta_src[:, :, graph.forward_index(graph.nodes['input'], False)].clone()

        # ---------- forward clean once ----------
        with model.hooks(fwd_hooks=ch_clean):
            clean_logits = model(clean_toks, attention_mask=attn_mask)

        # delta_src now has (corrupt - clean)
        delta_src_full = store.delta_src  # (B,P,F,H) requires_grad=False but depends on acts? we used no_grad for src adds, so set requires_grad?
        delta_src_full = delta_src_full.to(device, dtype)

        # integrated gradients loop over input node only
        cs = corrupted_input
        cl = cs - store.delta_src[:, :, graph.forward_index(graph.nodes['input'], False)]  # clean input

        batch_scores = torch.zeros(nF, nB, device=device, dtype=dtype, requires_grad=True)
        for k in range(steps):
            alpha = (k + 1) / steps
            interp = cs + alpha * (cl - cs)
            interp = interp.requires_grad_(True)

            # small hook to replace input node output
            def input_replace_hook(act, hook):
                return interp

            # recompute dst inputs each step because they depend on current forward
            # We'll need to rebuild dst list each iter
            # simpler: only care grads wrt dst inputs; intercept them again
            def dst_capture(prev_idx: int, bwd_idx: int):
                def fn(inp, hook):
                    inp = inp.requires_grad_(True)
                    return inp
                return fn

            # but we also need to know which tensors map to which dst index.
            # We'll re-create store2
            store2 = []
            def dst_record(prev_idx, bwd_idx):
                def fn(x, hook):
                    x = x.requires_grad_(True)
                    store2.append((x, (prev_idx, bwd_idx)))
                    return x
                return fn

            # build hooks
            step_hooks = [(graph.nodes['input'].out_hook, input_replace_hook)]
            # for dst nodes
            for layer in range(graph.cfg['n_layers']):
                a0 = graph.nodes[f'a{layer}.h0']
                prev = graph.prev_index(a0)
                for i,letter in enumerate('qkv'):
                    bwd = graph.backward_index(a0, qkv=letter, attn_slice=False)
                    step_hooks.append((a0.qkv_inputs[i], dst_record(prev, bwd)))
                m = graph.nodes[f'm{layer}']
                prev = graph.prev_index(m)
                bwd = graph.backward_index(m, attn_slice=False)
                step_hooks.append((m.in_hook, dst_record(prev,bwd)))
            ln = graph.nodes['logits']
            prev = graph.prev_index(ln)
            bwd  = graph.backward_index(ln, attn_slice=False)
            step_hooks.append((ln.in_hook, dst_record(prev,bwd)))

            logits = None
            with model.hooks(fwd_hooks=step_hooks):
                logits = model(clean_toks, attention_mask=attn_mask)

            metric_val = metric(logits, clean_logits, lengths, label).mean()
            grads = torch.autograd.grad(metric_val, [t for t,_ in store2], create_graph=True, retain_graph=True)
            # compute scores for this step
            for (g, (prev_i, bwd_i)) in zip(grads, [m for _,m in store2]):
                if prev_i == 0:
                    continue

                # 2) 取 Δh_src
                # 计算 contrib 处
                f_slice = delta_src_full[:, :, :prev_i]        # (B,P,F,Hs)
                g_aligned = g
                if g_aligned.ndim > 3:
                    g_aligned = g_aligned.flatten(-2, -1)
                Hs_src = f_slice.size(-1); Hs_dst = g_aligned.size(-1)
                if Hs_src != Hs_dst:
                    if Hs_dst % Hs_src == 0:
                        g_aligned = g_aligned.view(g_aligned.size(0), g_aligned.size(1), Hs_src, -1).mean(-1)
                    elif Hs_src % Hs_dst == 0:
                        f_slice = f_slice.view(f_slice.size(0), f_slice.size(1), f_slice.size(2), Hs_dst, -1).mean(-1)
                        Hs_src = Hs_dst
                    else:
                        continue  # 或者 raise

                CHUNK = 128
                for s in range(0, prev_i, CHUNK):
                    e = min(prev_i, s + CHUNK)
                    contrib_chunk = (f_slice[:, :, s:e] * g_aligned.unsqueeze(2)).sum(dim=(0,1,3))  # (chunk,)

                    upd = torch.zeros_like(batch_scores, requires_grad=False)
                    if isinstance(bwd_i, slice):
                        upd[s:e, bwd_i] = contrib_chunk.unsqueeze(1)
                    else:
                        upd[s:e, bwd_i] = contrib_chunk
                    batch_scores = batch_scores + upd

            total_steps += 1

        batch_scores = batch_scores / steps  # IG average
        if score_acc is None:
            score_acc = batch_scores
        else:
            score_acc = score_acc + batch_scores
        n_items += B

    scores = score_acc / n_items
    return scores  # requires_grad=True

# ---------------- Public API ----------------

def attribute_diff(
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor, Tensor, Tensor, List[str]], Tensor],
    method: Literal["EAP-IG-inputs"] = "EAP-IG-inputs",
    ig_steps: int = 30,
    quiet: bool = False,
) -> Tensor:
    if method != "EAP-IG-inputs":
        raise NotImplementedError("Only EAP-IG-inputs diff version implemented here")
    return eap_ig_inputs_diff(model, graph, dataloader, metric, steps=ig_steps, quiet=quiet)