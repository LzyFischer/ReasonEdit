import gc
from functools import partial
from typing import Callable, List, Union

import einops
import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
import os
import pdb

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from src.eap_graph import EAPGraph

os.environ["PYTORCH_SDP_BACKEND"] = "math"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def EAP_corrupted_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph
):
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        # We are in the case of a residual layer or MLP
        # Activations have shape [batch_size, seq_len, d_model]
        # We need to add an extra dimension to make it [batch_size, seq_len, 1, d_model]
        # The hook slice is a slice of length 1
        upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
    elif activations.ndim == 4:
        # We are in the case of an attention layer
        # Activations have shape [batch_size, seq_len, n_heads, d_model]
        upstream_activations_difference[:, :, hook_slice, :] = -activations

def EAP_clean_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph
):
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
    elif activations.ndim == 4:
        upstream_activations_difference[:, :, hook_slice, :] += activations

def _expand_kv_to_heads(result: torch.Tensor, model) -> torch.Tensor:
    """
    若 result 的“头维度”为 n_kv_heads，而 EAP 的累加目标按 n_heads 组织，
    则把该维度按组复制，扩展到 n_heads。否则原样返回。
    期望 result 的形状为 [..., H, ...]，其中 H 是头维度。
    """
    n_heads = getattr(model.cfg, "n_heads", None)
    n_kv_heads = getattr(model.cfg, "n_key_value_heads", n_heads)
    if n_heads is None or n_kv_heads is None:
        return result

    if n_kv_heads == n_heads:
        return result

    # 找到“头维度”位置：常见为倒数第二维（..., heads, d_head）
    # 也可能已经把 d_head 聚合掉，只剩 [..., heads]
    # 我们用启发式：优先尝试 dim=-2；若该维不是 n_kv_heads，再尝试最后一维 dim=-1。
    head_dim = None
    if result.dim() >= 2 and result.size(-2) == n_kv_heads:
        head_dim = -2
    elif result.size(-1) == n_kv_heads:
        head_dim = -1

    if head_dim is None:
        return result  # 看不出头维，保持安全退出

    repeat_factor = n_heads // n_kv_heads
    if repeat_factor * n_kv_heads != n_heads:
        # 非整除（极罕见），保守返回原值以免错累加
        return result

    return result.repeat_interleave(repeat_factor, dim=head_dim)

def EAP_clean_backward_hook(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph,
    model: HookedTransformer,
):
    hook_slice = graph.get_hook_slice(hook.name)

    # we get the slice of all upstream nodes that come before this downstream node
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
    # we want to multiply it by the upstream activations difference
    if grad.ndim == 3:
        grad_expanded = grad.unsqueeze(-2)  # Shape: [batch_size, seq_len, 1, d_model]
    else:
        grad_expanded = grad  # Shape: [batch_size, seq_len, n_heads, d_model]

    # we compute the mean over the batch_size and seq_len dimensions
    result = torch.matmul(
        upstream_activations_difference[:, :, earlier_upstream_nodes_slice],
        grad_expanded.transpose(-1, -2)
    ).sum(dim=0).sum(dim=0) # we sum over the batch_size and seq_len dimensions

    result = _expand_kv_to_heads(result, model)

    graph.eap_scores[earlier_upstream_nodes_slice, hook_slice] += result 

def EAP_downstream_patching_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph,
) -> Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]]:
    hook_slice = graph.downstream_hook_slice[hook.name]

    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # The tensor 'patch_difference' represents the sum of all upstream activation differences that are connected to this downstream node
    patch_difference = einops.einsum(
        graph.adj_matrix[earlier_upstream_nodes_slice, hook_slice],
        upstream_activations_difference[:, :, earlier_upstream_nodes_slice, :],
        "n_upstream n_downstream_at_hook, batch_size seq_len n_upstream d_model -> batch_size seq_len n_downstream_at_hook d_model"
    )

    # alternatively, it might be faster to
    # 1) gather the activation differences for every non-zero element in adj_matrix[:, hook_slice] and sum them
    # 2) use torch.sparse.mm to multiply the adj_matrix with the activation differences

    if activations.ndim == 3:
        assert patch_difference.shape[-2] == 1, "Number of downstream nodes should be 1 for this type of hook" 
        # patched_residual_stream = clean_residual_stream - (clean_upstream_activations - corrupted_upstream_activations)
        activations -= patch_difference.squeeze(-2)
    elif activations.ndim == 4:
        activations -= patch_difference
    
    return activations


def EAP(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch_size seq_len"],
    corrupted_tokens: Int[Tensor, "batch_size seq_len"],
    metric: Callable,
    upstream_nodes: List[str]=None,
    downstream_nodes: List[str]=None,
    batch_size: int=1,
):

    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)

    assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    num_prompts, seq_len = clean_tokens.shape[0], clean_tokens.shape[1]

    assert num_prompts % batch_size == 0, "Number of prompts must be divisible by batch size"

    upstream_activations_difference = torch.zeros(
        (batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
        requires_grad=False
    )

    # set the EAP scores to zero
    graph.reset_scores()

    upstream_hook_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_hook_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))

    corruped_upstream_hook_fn = partial(
        EAP_corrupted_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )

    clean_upstream_hook_fn = partial(
        EAP_clean_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )

    clean_downstream_hook_fn = partial(
        EAP_clean_backward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph,
        model=model
    )

    for idx in tqdm(range(0, num_prompts, batch_size)):
        # we first perform a forward pass on the corrupted input 
        model.add_hook(upstream_hook_filter, corruped_upstream_hook_fn, "fwd")

        # we don't need gradients for this forward pass
        # we'll take the gradients when we perform the forward pass on the clean input
        with torch.no_grad(): 
            corrupted_tokens = corrupted_tokens.to(model.cfg.device)
            model(corrupted_tokens[idx:idx+batch_size], return_type=None)        

        # now we perform a forward and backward pass on the clean input
        model.reset_hooks()
        model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
        model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")

        clean_tokens = clean_tokens.to(model.cfg.device)
        value = metric(model(clean_tokens[idx:idx+batch_size], return_type="logits"))
        value.backward()
        
        # We delete the activation differences tensor to free up memory
        model.zero_grad()
        upstream_activations_difference *= 0

    del upstream_activations_difference
    gc.collect()
    torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= num_prompts
    graph.eap_scores = graph.eap_scores.cpu()

    return graph
