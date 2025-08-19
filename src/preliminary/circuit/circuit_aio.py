from __future__ import annotations
from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR
#!/usr/bin/env python3
"""
generate_attr_scores.py
────────────────────────────────────────────────────────────────────────────
1. 为给定 DATA_FILE 中的 prompts 生成多个随机 A/B 子集
2. 对每个子集计算 mask_gradient_prune_scores
3. 按 TOP_QUANT 阈值剪枝并保存 JSON + 可视化
"""

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import List
import torch.nn as nn

import torch as t
from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.types import PruneScores


# ═════════════════════════════ 1. 工具函数 ════════════════════════════════
def model_tag(name: str) -> str:
    last = name.split("/")[-1]
    return "".join("_" if c in ".-" else c for c in last).lower()


def quant_tag(q: float) -> str:
    keep_pct = round((1.0 - q) * 100, 3)
    return ("%g" % keep_pct).replace(".", "_")


def load_model(model_name: str, device: t.device, args):
    # 1) 先按 auto_circuit 方式把“基础 HF 模型”加载进来
    base = load_tl_model(model_name, device)   # TL 包装的 HF 模型
    base.cfg.parallel_attn_mlp = False

    # 2) （可选）把 ckpt 权重灌进 base（在 patchable_model 之前做）
    if args.resume is not None:
        ckpt = t.load(args.resume, map_location="cpu")
        sd = ckpt.get("model_state", ckpt)  # 兼容直接存 state_dict 的情况

        # 将 ckpt 的张量 dtype 对齐到当前模型参数 dtype
        model_sd = base.state_dict()
        aligned_sd = {}
        for k, v in sd.items():
            if k in model_sd and t.is_tensor(v) and t.is_floating_point(v) and t.is_floating_point(model_sd[k]):
                aligned_sd[k] = v.to(model_sd[k].dtype)
            else:
                aligned_sd[k] = v

        missing, unexpected = base.load_state_dict(aligned_sd, strict=args.strict_resume)
        print(f"[ckpt] loaded from {args.resume}")
        print(f"        missing={len(missing)}  unexpected={len(unexpected)}")
        if "rng_state" in ckpt:
            t.set_rng_state(ckpt["rng_state"])

    # 3) 给每个 block 补上 hook（不影响已加载的权重）
    for i, block in enumerate(base.blocks):
        if not hasattr(block, "hook_resid_mid"):
            block.hook_resid_mid = nn.Identity()
            block.hook_resid_mid.name = f"blocks.{i}.hook_resid_mid"

    print("[info]  Added Identity hook_resid_mid to all blocks")

    # 4) 最后再做 patchable 包装
    return patchable_model(
        base, factorized=False, slice_output="last_seq",
        separate_qkv=True, device=device,
    )


def compute_scores(
    model, prompts: List[str], device, batch_size: int = 1
) -> PruneScores:
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
        json.dump({"prompts": prompts}, tf)
        tf.flush()
        tmp_path = Path(tf.name)

    loader, _ = load_datasets_from_json(
        model=model, path=tmp_path, device=device,
        prepend_bos=True, batch_size=batch_size,
        train_test_size=(len(prompts), 1),
    )
    return mask_gradient_prune_scores(
        model=model, dataloader=loader,
        official_edges=None, grad_function="logit",
        answer_function="avg_diff", mask_val=0.0,
    )


def prune_and_save(
    scores: PruneScores,
    model,
    out_dir: Path,
    logic_name: str,
    subset_tag: str,
    top_quants: List[float],
    run_tag: str,                      # <── NEW: tag inserted right before filename
):
    flat_pos = t.cat(
        [t.tensor(v).flatten() for v in scores.values()],
        dim=0,
    )

    for q in top_quants:
        thresh = flat_pos.quantile(q).item() if flat_pos.numel() else 0.0
        pruned = {k: (t.tensor(v) * (t.tensor(v) >= thresh).float())
                  for k, v in scores.items()}

        qtag   = quant_tag(q)
        subdir = out_dir / qtag / run_tag         # <── put tag right before file
        (subdir / "figures").mkdir(parents=True, exist_ok=True)

        json_path = subdir / f"{logic_name}_{subset_tag}.json"
        with open(json_path, "w") as fp:
            json.dump({k: v.tolist() for k, v in pruned.items()}, fp, indent=2)

        # 可视化（如该函数内部会保存，则也会落在当前工作目录或需另外指定）
        draw_seq_graph(
            model, pruned,
            score_threshold=thresh,
            layer_spacing=True, orientation="v",
            display_ipython=False,
        )
        print(f"   ✓ {subset_tag:<14} q={q:.3f} → {json_path.relative_to(out_dir)}")


# ═════════════════════════════ 2. CLI 入口 ════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                   help="HuggingFace model name / path")
    p.add_argument("--data_file", default=str(DATA_DIR / "corrupt/augmented_dataset.json"),
                   help="JSON file containing a list of prompt groups")
    p.add_argument("--quants", nargs="+", type=float, default=[0.95, 0.9],
                   help="Top-quantiles for pruning (space separated)")
    p.add_argument("--subset_k", type=int, default=20,
                   help="Size of partA / partB for each split")
    p.add_argument("--splits", type=int, default=1,
                   help="Number of random splits to generate")
    p.add_argument("--seed", type=int, default=0,
                   help="Base seed for reproducibility")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size (only forward+backward, small is fine)")
    p.add_argument("--out_root", default=str(OUTPUTS_DIR / "attr_scores"),
                   help="Root directory to save results")
    p.add_argument("--resume", type=Path,
                   help="Path to checkpoint (expects {'model_state': ...} or raw state_dict)")
    p.add_argument("--strict_resume", action="store_true",
                   help="Strict key match when loading state_dict")
    return p


def main():
    args = build_parser().parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model  = load_model(args.model, device, args)

    out_root = Path(args.out_root).resolve()
    run_tag  = Path(args.resume).stem if args.resume else "origin"
    print(f"[info] Using run tag: {run_tag}")

    model_dir = out_root / model_tag(args.model)
    # 预建各量化目录下的 tag/figures 方便后续保存
    for q in args.quants:
        (model_dir / quant_tag(q) / run_tag / "figures").mkdir(parents=True, exist_ok=True)

    with open(args.data_file) as f:
        logic_groups = json.load(f)

    for idx, logic in enumerate(logic_groups):
        prompts = logic.get("prompts", [])
        if not prompts:
            continue

        logic_name = f"logic_{idx:03d}"
        print(f"\n▶ {logic_name} — {len(prompts)} prompts")

        k = min(args.subset_k, len(prompts) // 2 or 1)

        for split in range(1, args.splits + 1):
            rng = random.Random(args.seed + split - 1)
            shuffled = prompts[:]
            rng.shuffle(shuffled)

            partA, partB = shuffled[:k], shuffled[k: 2 * k]
            scores = compute_scores(model, partA, device, args.batch_size)
            prune_and_save(
                scores, model, model_dir,
                logic_name, f"split{args.seed + split - 1}_partA",
                args.quants, run_tag=run_tag,
            )
            if partB:
                scores_B = compute_scores(model, partB, device, args.batch_size)
                prune_and_save(
                    scores_B, model, model_dir,
                    logic_name, f"split{args.seed + split - 1}_partB",
                    args.quants, run_tag=run_tag,
                )

    print("\n✅ Finished for all TOP_QUANTS.")


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()