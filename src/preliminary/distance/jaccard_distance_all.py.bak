#!/usr/bin/env python3
"""
jaccard_matrix.py
Compute per-logic Jaccard distance (avg. over 3 splits of partA/B)
and plot the heat-map.
"""
import argparse, json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import os, re, collections

# ──────────────────────────────────────────────────────────────────────────
def load_pos_edges(path: Path, thr: float = 1e-5) -> set[str]:
    """Flatten nested-list JSON → {edge_name | weight > thr}."""
    raw = json.loads(path.read_text())
    edges = set()

    def rec(name, arr):
        if isinstance(arr, (int, float)):
            if arr > thr:
                edges.add(name)
        else:
            for i, v in enumerate(arr):
                rec(f"{name}:{i}", v)

    for mod, arr in raw.items():
        rec(mod, arr)
    return edges


def jaccard(a: set, b: set) -> float:
    u = a | b
    return 0.0 if not u else 1.0 - len(a & b) / len(u)


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/100",
        help="Directory OR glob pattern with *split*_part[A|B].json files",
    )
    parser.add_argument("--block", type=int, default=3,
                        help="Grid line every N samples")
    args = parser.parse_args()
    args.output = os.path.join(args.input, "jaccard_matrix.png")

    # ----------------------------------------------------------------------
    in_path = Path(args.input)
    if any(ch in args.input for ch in "*?[]"):
        files = sorted(Path().glob(args.input))
    elif in_path.is_dir():
        files = sorted(in_path.rglob("logic_*_part*.json"))
    else:
        raise ValueError("`--input` must be a directory or glob pattern")

    if not files:
        raise RuntimeError("No JSON files found!")

    # ---- 将六个文件归并为 {logic_name: {split_id: (edgesA, edgesB)}} ----
    logic_map = collections.defaultdict(lambda: collections.defaultdict(dict))
    pat = re.compile(r"(logic_\d+)_split(\d)_(part[AB])\.json")

    for f in files:
        m = pat.match(f.stem)
        if not m:
            continue
        logic, split, part = m.groups()
        logic_map[logic][int(split)][part] = load_pos_edges(f)

    # ---- 计算每个 logic 的平均 Jaccard(partA, partB) ----
    logic_names, logic_vals = [], []
    for logic, splits in sorted(logic_map.items()):
        vals = []
        for s_id, parts in splits.items():
            if "partA" in parts and "partB" in parts:
                vals.append(jaccard(parts["partA"], parts["partB"]))
        if not vals:
            continue           # 若 split 缺文件则跳过
        logic_names.append(logic)
        logic_vals.append(np.mean(vals))

    n = len(logic_vals)
    if n == 0:
        raise RuntimeError("No complete partA/partB pairs found!")

    # ---- 构造对角矩阵便于沿用原热图代码 ----
    mat = np.zeros((n, n))
    for i, d in enumerate(logic_vals):
        mat[i, i] = d

    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, n * .4), max(5, n * .4)))
    im = ax.imshow(mat, cmap="viridis")
    block = max(1, int(args.block))
    for k in range(block, n, block):
        ax.axhline(k - 0.5, color="k", lw=0.5, ls="--")
        ax.axvline(k - 0.5, color="k", lw=0.5, ls="--")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(logic_names, rotation=90, fontsize=8)
    ax.set_yticklabels(logic_names, fontsize=8)
    ax.set_title("Avg. Jaccard Distance  (partA vs. partB, 3-split mean)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"✓ Saved heat-map → {out_path}")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()