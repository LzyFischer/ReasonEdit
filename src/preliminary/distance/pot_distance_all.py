#!/usr/bin/env python3
"""
pot_distance_matrix.py
Compute pair-wise Optimal-Transport (Wasserstein-1) distance
for circuit-mask JSON files and plot the heat-map.

USAGE
-----
python pot_distance_matrix.py \
    --input /scratch/.../output/attr_scores/qwen1_5_1_8b_chat/100
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ot  # pip install pot

# ──────────────────────────────────────────────────────────────────────
def load_edges(path: Path) -> list[str]:
    """Flatten nested-list JSON → list of positive-score edge names."""
    raw = json.loads(path.read_text())
    edges = []

    def rec(name, arr):
        if isinstance(arr, (int, float)):
            if arr > 0:
                edges.append(name)
        else:
            for i, v in enumerate(arr):
                rec(f"{name}:{i}", v)

    for mod, arr in raw.items():
        rec(mod, arr)
    return edges

# ──────────────────────────────────────────────────────────────────────
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

    # resolve files ---------------------------------------------------------
    in_path = Path(args.input)
    if any(c in args.input for c in "*?["):
        files = sorted(Path().glob(args.input))
        out_dir = Path(os.path.commonpath(files))
    elif in_path.is_dir():
        files = sorted(in_path.rglob("logic_*.json"))
        out_dir = in_path
    else:
        raise ValueError("`--input` must be a directory or glob pattern")

    if not files:
        raise RuntimeError("No JSON files found!")

    names = [f.stem for f in files]
    bags  = [load_edges(f) for f in files]

    # build vocabulary & histograms ----------------------------------------
    vocab  = {e: i for i, e in enumerate(sorted({e for bag in bags for e in bag}))}
    m      = len(vocab)
    if m == 0:
        raise RuntimeError("No positive edges in files.")

    def hist(bag):
        vec = np.bincount([vocab[e] for e in bag], minlength=m).astype(np.float64)
        return vec / max(1, vec.sum())      # probability distribution

    H = [hist(b) for b in bags]

    # ground cost matrix: 0 if same edge else 1 ----------------------------
    C = np.ones((m, m), dtype=np.float64) - np.eye(m)

    # OT distance matrix ----------------------------------------------------
    n   = len(H)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = ot.emd2(H[i], H[j], C)    # Wasserstein-1 (squared cost)

    # plot ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, n * .4), max(5, n * .4)))
    im      = ax.imshow(mat, cmap="magma")

    blk = max(1, int(args.block))
    for k in range(blk, n, blk):
        ax.axhline(k - 0.5, color="w", lw=1.0, ls="--")
        ax.axvline(k - 0.5, color="w", lw=1.0, ls="--")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Pairwise Optimal-Transport Distance")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_png = out_dir / "ot_distance_matrix.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"✓ Saved heat-map → {out_png}")

# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()