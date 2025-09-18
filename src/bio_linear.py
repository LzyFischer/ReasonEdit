
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class BioLinear1D(nn.Module):
    # A drop-in replacement for nn.Linear with 1D neuron coordinates and CC regularizer.
    def __init__(self, in_features: int, out_features: int, bias: bool = True, in_fold: int = 1, out_fold: int = 1, l0: float = 0.1) -> None:
        super().__init__()
        assert in_features % in_fold == 0, "in_features must be divisible by in_fold"
        assert out_features % out_fold == 0, "out_features must be divisible by out_fold"
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_fold = in_fold
        self.out_fold = out_fold
        self.l0 = l0

        in_dim_fold = in_features // in_fold
        out_dim_fold = out_features // out_fold
        in_coords_one = torch.linspace(1/(2*in_dim_fold), 1-1/(2*in_dim_fold), steps=in_dim_fold)
        out_coords_one = torch.linspace(1/(2*out_dim_fold), 1-1/(2*out_dim_fold), steps=out_dim_fold)
        self.register_buffer("in_coordinates", in_coords_one.repeat(in_fold))
        self.register_buffer("out_coordinates", out_coords_one.repeat(out_fold))

        self.input_cache = None
        self.output_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_cache = x
        y = self.linear(x)
        self.output_cache = y
        return y

    @torch.no_grad()
    def _swap_weight(self, j: int, k: int, swap_type: str = "out") -> None:
        W = self.linear.weight
        if swap_type == "out":
            tmp = W[j].clone()
            W[j] = W[k]
            W[k] = tmp
        elif swap_type == "in":
            tmp = W[:, j].clone()
            W[:, j] = W[:, k]
            W[:, k] = tmp
        else:
            raise ValueError(f"Unknown swap_type {swap_type!r}")

    @torch.no_grad()
    def _swap_bias(self, j: int, k: int) -> None:
        if self.linear.bias is None:
            return
        b = self.linear.bias
        tmp = b[j].clone()
        b[j] = b[k]
        b[k] = tmp

    @torch.no_grad()
    def relocate_single(self, layer_side: str, j: int) -> None:
        if layer_side not in {"in", "out"}:
            raise ValueError("layer_side must be 'in' or 'out'")
        num = self.in_features if layer_side == "in" else self.out_features
        best_cc = float("inf")
        best_k = j
        for k in range(num):
            if k == j:
                cc0 = self.connection_cost(weight_factor=1.0, bias_penalize=True, no_penalize_this=False)
                best_cc = float(cc0.item())
                continue
            if layer_side == "in":
                self._swap_weight(j, k, swap_type="in")
            else:
                self._swap_weight(j, k, swap_type="out")
                self._swap_bias(j, k)
            cc = self.connection_cost(weight_factor=1.0, bias_penalize=True, no_penalize_this=False)
            if layer_side == "in":
                self._swap_weight(j, k, swap_type="in")
            else:
                self._swap_weight(j, k, swap_type="out")
                self._swap_bias(j, k)
            if float(cc.item()) < best_cc:
                best_cc = float(cc.item())
                best_k = k
        if best_k != j:
            if layer_side == "in":
                self._swap_weight(j, best_k, swap_type="in")
            else:
                self._swap_weight(j, best_k, swap_type="out")
                self._swap_bias(j, best_k)

    @torch.no_grad()
    def get_top_indices(self, layer_side: str, top_k: int = 32) -> torch.Tensor:
        W = self.linear.weight
        if layer_side == "in":
            score = W.abs().sum(dim=0)
        elif layer_side == "out":
            score = W.abs().sum(dim=1)
        else:
            raise ValueError("layer_side must be 'in' or 'out'")
        top = torch.argsort(score, descending=True)[:top_k]
        return top

    @torch.no_grad()
    def relocate_topk(self, layer_side: str, top_k: int = 32) -> None:
        top = self.get_top_indices(layer_side, top_k=top_k)
        for j in top.tolist():
            self.relocate_single(layer_side, j)

    def connection_cost(self, weight_factor: float = 1.0, bias_penalize: bool = True, no_penalize_this: bool = False) -> torch.Tensor:
        if no_penalize_this:
            return torch.zeros((), device=self.linear.weight.device)
        W = self.linear.weight
        in_coords = self.in_coordinates.to(W.device)
        out_coords = self.out_coordinates.to(W.device)
        dist = (out_coords[:, None] - in_coords[None, :]).abs()
        cc = (W.abs() * (weight_factor * dist + self.l0)).sum()
        if bias_penalize and self.linear.bias is not None:
            cc = cc + (self.linear.bias.abs() * self.l0).sum()
        return cc
