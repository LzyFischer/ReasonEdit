#!/usr/bin/env python3
"""
weighted_edit_distance_matrix.py
Compute pair-wise weighted edit distance (∑|p_i-q_i|) on **normalised** weights
and plot the heat-map.

运行示例
--------
python weighted_edit_distance_matrix.py \
    --input /scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/1
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
def load_weight_dict(path: Path, mode: str = "sum") -> dict[str, float]:
    """
    JSON -> {edge : normalised_weight}
    mode:
      "sum":  weight / Σweight   (概率分布)
      "max":  weight / max(weight)
    """
    raw = json.loads(path.read_text())
    flat = {}

    def rec(name, arr):
        if isinstance(arr, (int, float)):
            if arr > 0:
                flat[name] = float(arr)
        else:
            for i, v in enumerate(arr):
                rec(f"{name}:{i}", v)

    for mod, arr in raw.items():
        rec(mod, arr)

    if not flat:
        return flat

    if mode == "sum":
        s = sum(flat.values())
        flat = {k: v / s for k, v in flat.items()}
    elif mode == "max":
        m = max(flat.values())
        flat = {k: v / m for k, v in flat.items()}

    return flat


def w_edit(d1: dict[str, float], d2: dict[str, float]) -> float:
    """L-1 distance on the union of keys (∑ |p_i - q_i|)."""
    keys = set(d1) | set(d2)
    return sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)
    # 若想限制到 [0,1]，可以再除以 2.


# ────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/1",
        help="Directory OR glob pattern of *.json mask files"
    )
    parser.add_argument(
        "--block", type=int, default=3,
        help="Draw dotted grid line every N samples (default: 3)"
    )
    args = parser.parse_args()

    # 解析输入文件 -----------------------------------------------------------
    in_path = Path(args.input)
    if any(ch in args.input for ch in "*?["):
        files = sorted(Path().glob(args.input))
        out_dir = Path(os.path.commonpath(files))  # glob -> 取共同父目录
    elif in_path.is_dir():
        files = sorted(in_path.rglob("logic_*.json"))
        out_dir = in_path
    else:
        raise ValueError("`--input` must be a directory or glob pattern")

    if not files:
        raise RuntimeError("No JSON files found!")

    names  = [f.stem for f in files]
    dicts  = [load_weight_dict(f, mode="sum") for f in files]

    # 距离矩阵 ---------------------------------------------------------------
    n   = len(dicts)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = w_edit(dicts[i], dicts[j])

    # 绘图 -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, n * .4), max(5, n * .4)))
    im      = ax.imshow(mat, cmap="cividis")
    blk     = max(1, int(args.block))

    for k in range(blk, n, blk):
        ax.axhline(k - 0.5, color="w", lw=1.0, ls="--")
        ax.axvline(k - 0.5, color="w", lw=1.0, ls="--")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Pairwise Weighted-Edit Distance (normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_png = out_dir / "weighted_edit_distance_matrix.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"✓ Saved heat-map → {out_png}")

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()