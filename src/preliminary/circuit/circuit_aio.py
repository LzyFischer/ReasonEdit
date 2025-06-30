from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR
#!/usr/bin/env python3
"""
generate_attr_scores.py
────────────────────────────────────────────────────────────────────────────
1. 为给定 DATA_FILE 中的 prompts 生成多个随机 A/B 子集
2. 对每个子集计算 mask_gradient_prune_scores
3. 按 TOP_QUANT 阈值剪枝并保存 JSON + 可视化
"""

from __future__ import annotations
import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import List
import torch.nn as nn

import pdb
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


def load_model(model_name: str, device: t.device):
    base = load_tl_model(model_name, device)
    base.cfg.parallel_attn_mlp = False
    for i, block in enumerate(base.blocks):
        if not hasattr(block, "hook_resid_mid"):
            block.hook_resid_mid = nn.Identity()
            # 让它在 state_dict 里有名字，避免再被找不到
            block.hook_resid_mid.name = f"blocks.{i}.hook_resid_mid"
    print("[info]  Added Identity hook_resid_mid to all blocks")
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
):
    # flat_pos = t.cat(
    #     [t.tensor(v).flatten()[t.tensor(v).flatten() > 0.0]
    #      for v in scores.values()],
    #     dim=0,
    # )
    flat_pos = t.cat(
        [t.tensor(v).flatten()
         for v in scores.values()],
        dim=0,
    )

    for q in top_quants:
        thresh = flat_pos.quantile(q).item() if flat_pos.numel() else 0.0
        pruned = {k: (t.tensor(v) * (t.tensor(v) >= thresh).float())
                  for k, v in scores.items()}


        qtag   = quant_tag(q)
        subdir = out_dir / qtag
        subdir.mkdir(parents=True, exist_ok=True)

        json_path = subdir / f"{logic_name}_{subset_tag}.json"
        with open(json_path, "w") as fp:
            json.dump({k: v.tolist() for k, v in pruned.items()}, fp, indent=2)
        
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
    return p


def main():
    args = build_parser().parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    model  = load_model(args.model, device)

    model_dir = Path(args.out_root) / model_tag(args.model)
    for q in args.quants:
        (model_dir / quant_tag(q) / "figures").mkdir(parents=True, exist_ok=True)

    with open(args.data_file) as f:
        logic_groups = json.load(f)

    for idx, logic in enumerate(logic_groups):
        prompts = logic.get("prompts", [])
        if not prompts or idx < 3:
            continue

        logic_name = f"logic_{idx:03d}"
        print(f"\n▶ {logic_name} — {len(prompts)} prompts")

        k = min(args.subset_k, len(prompts) // 2 or 1)

        for split in range(1, args.splits + 1):
            rng = random.Random(args.seed + split)
            shuffled = prompts[:]
            rng.shuffle(shuffled)

            partA, partB = shuffled[:k], shuffled[k: 2 * k]
            scores = compute_scores(model, partA, device, args.batch_size)
            prune_and_save(
                scores, model, model_dir,
                logic_name, f"split{args.seed + split}_partA",
                args.quants,
            )

            if partB:
                scores_B = compute_scores(model, partB, device, args.batch_size)
                prune_and_save(
                    scores_B, model, model_dir,
                    logic_name, f"split{args.seed + split}_partB",
                    args.quants,
                )

    print("\n✅ Finished for all TOP_QUANTS.")


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()