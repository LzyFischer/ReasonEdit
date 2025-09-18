#!/usr/bin/env python3
import os
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PYTORCH_SDP_BACKEND"] = "math"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import pdb

# ------------------------------- CONFIG ---------------------------------------
MODEL_NAME   = "Qwen/Qwen2.5-3B-Instruct"   # or any HF CausalLM
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.bfloat16 if torch.cuda.is_available() else torch.float32
TOPK_PRINT   = 10   # how many params with grad to print

CLEAN_TEXT   = [
    "The soprano sang the aria flawlessly during the opera performance. However, the tenor did not forgot his lines on stage.\nIf the soprano sang flawlessly or the tenor forgot his lines, then it implies the audience would applaud the overall performance.\nDid the audience end up applauding the opera performance? (Answer in True, False, or N/A (Neither)). Answer:",
    "The soprano sang the aria flawlessly not the opera performance. However, the tenor did not forgot his lines on stage.\nIf the soprano sang flawlessly or the tenor forgot his lines, then it implies the audience would applaud the overall performance.\nDid the audience end up applauding the opera performance? (Answer in True, False, or N/A (Neither)). Answer:"
]
CORR_TEXT    = [
    "The soprano sang the aria flawlessly not the opera performance. However, the tenor did not forgot his lines on stage.\nIf the soprano sang flawlessly or the tenor forgot his lines, then it implies the audience would applaud the overall performance.\nDid the audience end up applauding the opera performance? (Answer in True, False, or N/A (Neither)). Answer:",
    "The soprano sang the aria flawlessly not the opera performance. However, the tenor did not forgot his lines on stage.\nIf the soprano sang flawlessly or the tenor forgot his lines, then it implies the audience would applaud the overall performance.\nDid the audience end up applauding the opera performance? (Answer in True, False, or N/A (Neither)). Answer:"
]
ANSWER       = " True"   # NOTE: leading space if your tokenizer splits like that

# -----------------------------UTILS-----------------------------------
def report_param_grads(model: torch.nn.Module, topk: int = 10) -> Tuple[int, List[Tuple[str, float]]]:
    """
    统计参数梯度并打印 Top-K。
    Returns:
        nonzero (int): 有非零梯度的参数个数
        top_items (List[Tuple[str, float]]): [(param_name, grad_abs_sum), ...]
    """
    nonzero = 0
    top_items = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            s = p.grad.abs().sum().item()
            if s > 0:
                nonzero += 1
                top_items.append((name, s))
    top_items = sorted(top_items, key=lambda x: x[1], reverse=True)[:topk]

    print(f"[INFO] Params with non-zero grad from effect-loss: {nonzero}")
    for n, s in top_items:
        print(f"  • {n}: grad_abs_sum={s:.3e}")

    return nonzero, top_items

def report_effects(
    effects: Dict[str, torch.Tensor],
    topk_node: int = 15,
    topk_head: int = 40
) -> Tuple[Dict[str, float], List[Tuple[str, int, float]], List[Tuple[str, float]], List[Tuple[str, int, float]]]:
    """
    对每个节点/每个 head 的 effect 做统计并打印 Top-K。

    effects:
        - q/k/v: 形状 [B, n_heads] 或 [B, n_kv_heads] (可能扩展到 n_heads)
        - 其他: 形状 [B]

    Returns:
        node_scores: {node_name: scalar_score}
        head_scores: [(node_name, head_idx, score)]
        top_nodes  : 前 topk_node 节点 [(node_name, score)]
        top_heads  : 前 topk_head 头   [(node_name, head_idx, score)]
    """
    node_scores = {}
    head_scores = []

    for n, v in effects.items():
        if v.dim() == 2:
            # [B, n_heads]
            score_per_head = v.abs().mean(dim=0)  # [n_heads]
            for h, s in enumerate(score_per_head.tolist()):
                head_scores.append((n, h, s))
            node_scores[n] = score_per_head.mean().item()
        elif v.dim() == 1:
            # [B]
            node_scores[n] = v.abs().mean().item()
        else:
            node_scores[n] = v.abs().mean().item()

    top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:topk_node]
    print("\n[TOP NODE EFFECTS]")
    for name, sc in top_nodes:
        print(f"  {sc:9.3e}  {name}")

    head_scores = sorted(head_scores, key=lambda x: x[2], reverse=True)[:topk_head]
    print("\n[TOP HEAD EFFECTS]  (node, head_idx, score)")
    for n, h, sc in head_scores:
        print(f"  {sc:9.3e}  {n}  head={h}")

    return node_scores, head_scores, top_nodes, head_scores[:topk_head]


# ----------------------------- HOOK HELPERS -----------------------------------
class ActCacher:
    """Caches forward activations for a given set of module names."""
    def __init__(self, model, names):
        self.model = model
        self.names = names
        self.cache = {}
        self.hooks = []

    def _hook(self, name):
        def fn(module, inp, out):
            self.cache[name] = out
        return fn

    def __enter__(self):
        for n in self.names:
            m = self.model.get_submodule(n)
            self.hooks.append(m.register_forward_hook(self._hook(n)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.hooks: h.remove()
        self.hooks = []

# ----------------------------- NODE SELECTOR ----------------------------------
def get_comp_nodes(model):
    """
    Pick nodes you want 'effects' for.
    Example: all attention projections & MLP outputs (excluding container modules).
    """
    names = []
    for name, module in model.named_modules():
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"]):
            names.append(name)
    return names

# ----------------------------- METRIC FN --------------------------------------
def token_logit_metric(outputs, tokenizer, answers):
    """
    Metric: mean logit of the target answer token at last position.
    """
    logits_last = outputs.logits[:, -1, :]  # [B, vocab]
    if isinstance(answers, str):
        answers = [answers] * logits_last.size(0)
    tgt_ids = []
    for a in answers:
        ids = tokenizer.encode(a, add_special_tokens=False)
        if len(ids) == 0:
            ids = [tokenizer.eos_token_id]
        tgt_ids.append(ids[0])
    tgt_ids = torch.tensor(tgt_ids, device=logits_last.device)
    chosen = logits_last.gather(1, tgt_ids.unsqueeze(1)).squeeze(1)  # [B]
    return chosen.mean()  # scalar

def view_heads(t, n_heads):
    # t: [B, hidden]  ->  [B,  n_heads, head_dim]
    B, H = t.shape
    head_dim = H // n_heads
    return t.view(B, n_heads, head_dim)

def per_head_effect(diff, grad):
    # diff/grad: [B, n_heads, head_dim]
    return (diff * grad).mean(-1)   # [B, n_heads]

# ----------------------------- EFFECT + GRAD ----------------------------------
@torch.enable_grad()
def calculate_effect(
    model: nn.Module,
    clean_cache: "ActCacher",
    corrupt_cache: "ActCacher",
    nodes: List[str],
    tokenizer: AutoTokenizer,
    out_clean,
    answer: str,
    last_token_only: bool = True,
    return_debug: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    """
    返回:
      effects: node -> Tensor  (q/k/v: [B, n_heads or n_kv_heads], others: [B])
      act_grads: node -> Tensor (激活在最后 token 的梯度，形状 [B, hidden])
      debug_cache: node -> {"diff":..., "grad":..., "eff":...}（当 return_debug=True）
    """
    n_heads    = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
    group_size = max(1, n_heads // max(1, n_kv_heads))

    metric = token_logit_metric(out_clean, tokenizer, answer)

    # 对每个节点的 clean 激活，求 metric 对激活的梯度
    grads = torch.autograd.grad(
        metric,
        [clean_cache.cache[n] for n in nodes],
        create_graph=True,
        allow_unused=True
    )

    effects: Dict[str, torch.Tensor] = {}
    act_grads: Dict[str, torch.Tensor] = {}
    debug_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    for n, g in zip(nodes, grads):
        if g is None:
            continue

        act_clean = clean_cache.cache[n]   # [B, seq, hidden] or other
        act_cor   = corrupt_cache.cache[n]
        diff_full = act_clean - act_cor    # [B, seq, hidden]

        if last_token_only:
            diff_last = diff_full[:, -1]       # [B, hidden]
            grad_last = g[:, -1]               # [B, hidden]
        else:
            # 展平所有 token
            B, S, H = diff_full.shape
            diff_last = diff_full.reshape(B, S*H)
            grad_last = g.reshape(B, S*H)

        leaf = n.split(".")[-1]
        if leaf in ("q_proj", "k_proj", "v_proj"):
            if leaf == "q_proj":
                # q 使用全部 n_heads
                diff_h = view_heads(diff_last, n_heads)    # [B, n_heads, head_dim]
                grad_h = view_heads(grad_last, n_heads)    # [B, n_heads, head_dim]
                eff_h  = per_head_effect(diff_h, grad_h)   # [B, n_heads]
                effects[n]   = eff_h
                act_grads[n] = grad_last                   # 原始激活梯度（[B, hidden]）
                if return_debug:
                    debug_cache[n] = {"diff": diff_h, "grad": grad_h, "eff": eff_h}
            else:
                # k/v 使用 n_kv_heads；如需与 q 对齐，可 repeat_interleave 到 n_heads
                diff_kv = view_heads(diff_last, n_kv_heads)  # [B, n_kv_heads, head_dim]
                grad_kv = view_heads(grad_last, n_kv_heads)  # [B, n_kv_heads, head_dim]
                eff_kv  = per_head_effect(diff_kv, grad_kv)  # [B, n_kv_heads]
                effects[n]   = eff_kv
                act_grads[n] = grad_last
                if return_debug:
                    debug_cache[n] = {"diff": diff_kv, "grad": grad_kv, "eff": eff_kv}
        else:
            # 其他层整体 effect：对 hidden 维平均
            eff = (diff_last * grad_last).reshape(diff_last.size(0), -1).mean(dim=1)  # [B]
            effects[n]   = eff
            act_grads[n] = grad_last
            if return_debug:
                debug_cache[n] = {"diff": diff_last, "grad": grad_last, "eff": eff}

    if return_debug:
        return effects, act_grads, debug_cache
    else:
        return effects, act_grads, {}

# ----------------------------- MAIN TEST --------------------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        attn_implementation="eager",
        device_map=None
    ).to(DEVICE)

    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None

    nodes = get_comp_nodes(model)
    print(f"[INFO] tracking {len(nodes)} nodes")

    # 1) Forward clean & corrupt, cache activations
    with ActCacher(model, nodes) as clean_cache:
        inputs_clean = tokenizer(CLEAN_TEXT, return_tensors="pt").to(DEVICE)
        out_clean    = model(**inputs_clean)

    with ActCacher(model, nodes) as corrupt_cache:
        inputs_cor   = tokenizer(CORR_TEXT, return_tensors="pt").to(DEVICE)
        _            = model(**inputs_cor)

    # 2) 直接用 calculate_effect 同时拿到 effect 和 activation-grad
    effects, act_grads, debug_cache = calculate_effect(
        model, clean_cache, corrupt_cache, nodes, tokenizer, out_clean, ANSWER, return_debug=True
    )

    # 3) 定义一个简单的 effect 损失（示例）
    if len(effects) == 0:
        raise RuntimeError("No effects were computed (all grads were None). Check node list.")
    loss_on_effects = torch.stack([v.mean() for v in effects.values()]).pow(2).mean()

    # 4) 反传到参数
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    torch.autograd.set_detect_anomaly(True)
    loss_on_effects.backward()
    optimizer.step()

    report_param_grads(model, topk=TOPK_PRINT)
    report_effects(effects, topk_node=15, topk_head=40)

    # Optional: sanity prints & interactive
    some_node = next(iter(effects))
    print(f"[DEBUG] one effect sample -> {some_node}: {effects[some_node].detach().cpu().numpy()}")
    # 你也可以用 act_grads[some_node] 做进一步分析 / 保护子空间等
    # pdb.set_trace()

if __name__ == "__main__":
    main()