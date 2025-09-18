
import torch
import torch.nn as nn
from typing import Iterable, Callable, Optional

# Reuse the BioLinear1D from your local copy (save alongside this file).
from src.bio_linear import BioLinear1D

def fold_params_for_name(name: str, n_heads: int):
    """
    Heuristic: if a Linear is a q/k/v projection, use out_fold=n_heads.
               if it's an o_proj, use in_fold=n_heads.
    Works for many HF LLMs (Llama/Qwen/etc.) that name submodules q_proj/k_proj/v_proj/o_proj.
    """
    name_lower = name.lower()
    if any(x in name_lower for x in ["q_proj", "k_proj", "v_proj"]):
        return dict(out_fold=n_heads)
    if "o_proj" in name_lower:
        return dict(in_fold=n_heads)
    return {}

def replace_linear_with_biolinear(module: nn.Module, n_heads: int, l0: float = 0.1):
    """
    Recursively replace all nn.Linear modules with BioLinear1D.
    Apply head-aware folding when names indicate attention projections.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            kwargs = fold_params_for_name(name, n_heads)
            new = BioLinear1D(child.in_features, child.out_features, bias=(child.bias is not None), l0=l0, **kwargs)
            new.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new.linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new)
        else:
            replace_linear_with_biolinear(child, n_heads=n_heads, l0=l0)

def iter_biolinears(module: nn.Module):
    for m in module.modules():
        if isinstance(m, BioLinear1D):
            yield m

def connection_cost(module: nn.Module, weight_factor: float = 1.0, bias_penalize: bool = True) -> torch.Tensor:
    cc = None
    for m in iter_biolinears(module):
        term = m.connection_cost(weight_factor=weight_factor, bias_penalize=bias_penalize, no_penalize_this=False)
        cc = term if cc is None else (cc + term)
    if cc is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return cc

@torch.no_grad()
def relocate_topk_every(module: nn.Module, top_k: int = 16, sides=("in","out")):
    """
    Perform greedy relocation on top-k important neurons per BioLinear layer.
    Use sparingly on big models.
    """
    for m in iter_biolinears(module):
        for side in sides:
            m.relocate_topk(layer_side=side, top_k=top_k)
