#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM


# =========================
# 1) CLI
# =========================
def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Checkpoint sanity & diff tester")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                   help="HF model name or local path")
    p.add_argument("--resume", type=str, default="ckpts/reptile_00001.pt",
                   help="Path to reptile_*.pt (or raw state_dict) checkpoint")
    p.add_argument("--strict_resume", type=str2bool, default=False,
                   help="strict=True requires exact key match")
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float16", "bfloat16", "float32"],
                   help="dtype for loading the base model")
    p.add_argument("--device", type=str, default="auto",
                   help='"auto"→cuda if available else cpu; or explicitly "cpu"/"cuda"')
    p.add_argument("--topk", type=int, default=20,
                   help="Print top-K tensors by max|Δ| in diff")
    return p.parse_args()


# =========================
# 2) Test utilities (your code + small tweaks)
# =========================
def extrema_of_state_dict(sd, max_examples=0):
    gmin = (float("inf"), None)   # (value, tensor_name)
    gmax = (float("-inf"), None)
    nonfinite_names = []

    for name, t in sd.items():
        if not torch.is_tensor(t) or not torch.is_floating_point(t):
            continue
        x = t.detach().to("cpu")
        finite = torch.isfinite(x)
        if not finite.any():
            nonfinite_names.append(name)
            continue
        xf = x[finite]
        tmin = xf.min().item()
        tmax = xf.max().item()
        if tmin < gmin[0]: gmin = (tmin, name)
        if tmax > gmax[0]: gmax = (tmax, name)

    print(f"[CKPT] global min = {gmin[0]:.6g}  (tensor: {gmin[1]})")
    print(f"[CKPT] global max = {gmax[0]:.6g}  (tensor: {gmax[1]})")
    if nonfinite_names:
        print(f"[CKPT] tensors with no finite values: {len(nonfinite_names)}")
        if max_examples:
            print("  e.g.:", ", ".join(nonfinite_names[:max_examples]))

def scan_state_dict_for_nonfinite(state_dict, max_report=10):
    """Return (has_issue, summary_str). Prints up to max_report offending tensors."""
    issues = []
    total_params = 0
    bad_params = 0

    for name, t in state_dict.items():
        if not torch.is_tensor(t) or not torch.is_floating_point(t):
            continue
        total_params += 1
        nan_ct = torch.isnan(t).sum().item()
        pinf_ct = torch.isposinf(t).sum().item()
        ninf_ct = torch.isneginf(t).sum().item()
        if nan_ct or pinf_ct or ninf_ct:
            bad_params += 1
            if len(issues) < max_report:
                with torch.no_grad():
                    try:
                        vmax = t.abs().max().item()
                    except RuntimeError:
                        vmax = float("nan")
                issues.append(
                    f"- {name}  dtype={t.dtype}  shape={tuple(t.shape)}  "
                    f"nan={nan_ct} +inf={pinf_ct} -inf={ninf_ct}  |param|_max≈{vmax}"
                )

    header = (f"[CKPT SCAN] checked {total_params} float tensors; "
              f"{bad_params} had non-finite values.")
    details = "\n".join(issues)
    return (bad_params > 0), (header + ("\n" + details if details else ""))

def scan_model_for_nonfinite(model, max_report=10):
    issues = []
    for n, p in model.named_parameters():
        if p.requires_grad and torch.is_floating_point(p):
            if not torch.isfinite(p).all():
                nan_ct = torch.isnan(p).sum().item()
                pinf_ct = torch.isposinf(p).sum().item()
                ninf_ct = torch.isneginf(p).sum().item()
                issues.append(f"- {n}: nan={nan_ct} +inf={pinf_ct} -inf={ninf_ct}")
                if len(issues) >= max_report: break
    if issues:
        raise ValueError("[POST-LOAD] Non-finite values found:\n" + "\n".join(issues))

def is_lora_param(name: str) -> bool:
    n = name.lower()
    return ("lora_" in n) or (".lora_a" in n) or (".lora_b" in n)

@torch.no_grad()
def diff_model_vs_state_dict(model, sd: Dict[str, torch.Tensor],
                             *, include_lora: bool = False,
                             topk: int = 15, atol: float = 0.0) -> None:
    """
    比较 model 当前参数 与 sd(state_dict) 的差异（逐张量），打印汇总与Top-K。
    只比较浮点张量；按 float32 进行差异计算。
    """
    base = model

    total_tensors = 0
    matched_tensors = 0
    total_elems = 0
    l2_sum_sq = 0.0  # 用于整体 L2
    max_abs_global: Tuple[float, str] = (0.0, "")
    top = []  # [(maxabs, meanabs, l2, numel, name, shape)]

    missing_keys = []
    shape_mismatch = []
    skipped_lora = 0

    for name, p in base.named_parameters():
        if not torch.is_floating_point(p):
            continue
        if (not include_lora) and is_lora_param(name):
            skipped_lora += 1
            continue

        total_tensors += 1
        if name not in sd:
            missing_keys.append(name)
            continue

        t_ckpt = sd[name]
        if not torch.is_tensor(t_ckpt) or not torch.is_floating_point(t_ckpt):
            continue

        if p.shape != t_ckpt.shape:
            shape_mismatch.append((name, tuple(p.shape), tuple(t_ckpt.shape)))
            continue

        matched_tensors += 1
        x = p.detach().to("cpu", dtype=torch.float32)
        y = t_ckpt.detach().to("cpu", dtype=torch.float32)

        diff = (x - y)
        mad = diff.abs().max().item()
        meanad = diff.abs().mean().item()
        l2 = torch.linalg.vector_norm(diff).item()
        n = diff.numel()

        total_elems += n
        l2_sum_sq += (l2 ** 2)

        if mad > max_abs_global[0]:
            max_abs_global = (mad, name)

        top.append((mad, meanad, l2, n, name, tuple(x.shape)))

    # 排序并打印
    top.sort(key=lambda z: z[0], reverse=True)
    overall_l2 = (l2_sum_sq ** 0.5)
    overall_rmse = (overall_l2 / max(total_elems, 1)) ** 0.5

    print("\n[DIFF] compare model params vs. state_dict")
    print(f"  tensors_checked={total_tensors}  matched={matched_tensors}  skipped_lora={skipped_lora}")
    print(f"  missing_keys={len(missing_keys)}  shape_mismatch={len(shape_mismatch)}")
    print(f"  overall_L2={overall_l2:.6g}  overall_RMSE={overall_rmse:.6g}")
    print(f"  global_max_abs={max_abs_global[0]:.6g}  @ {max_abs_global[1]}")

    if missing_keys:
        print(f"  e.g. missing: {missing_keys[:5]}")
    if shape_mismatch:
        show = [f"{n} {s0}->{s1}" for (n, s0, s1) in shape_mismatch[:5]]
        print(f"  e.g. shape mismatch: {show}")

    k = min(topk, len(top))
    if k > 0:
        print(f"\n  Top-{k} tensors by max|Δ|:")
        for i in range(k):
            mad, meanad, l2, n, name, shape = top[i]
            print(f"   {i+1:>2}. {name:60s}  {shape}  "
                  f"max|Δ|={mad:.6g}  mean|Δ|={meanad:.6g}  L2={l2:.6g}  numel={n}")


# =========================
# 3) Main
# =========================
def main():
    args = get_args()
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"[INFO] loading base model: {args.model_name}  device={device}  dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)

    print(f"[INFO] loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu")
    sd = ckpt.get("model_state", ckpt)  # 支持 raw state_dict

    # ---- 预检查 ckpt ----
    extrema_of_state_dict(sd, max_examples=5)
    has_bad, report = scan_state_dict_for_nonfinite(sd, max_report=20)
    print(report)
    if has_bad:
        print("[WARN] checkpoint contains NaN/Inf — you may want to abort or sanitize.")

    # ---- 预对比：当前模型 vs ckpt ----
    print("\n[PRE-LOAD] 参数与 ckpt 的差异：")
    diff_model_vs_state_dict(model, sd, include_lora=False, topk=args.topk)

    # ---- 加载 ----
    base = model
    missing, unexpected = base.load_state_dict(sd, strict=args.strict_resume)
    print(f"\n[LOAD] missing={len(missing)}  unexpected={len(unexpected)}")

    # ---- 后检查：模型数值健康 ----
    scan_model_for_nonfinite(base)

    # ---- 后对比：应接近0 ----
    print("\n[POST-LOAD] 参数与 ckpt 的差异（应当接近 0）：")
    diff_model_vs_state_dict(model, sd, include_lora=False, topk=args.topk)

    # 额外信息
    if "iter" in ckpt:
        print(f"[INFO] ckpt meta: iter={ckpt['iter']}")
    if "rng_state" in ckpt:
        print("[INFO] ckpt includes RNG state")

if __name__ == "__main__":
    main()