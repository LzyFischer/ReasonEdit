#!/usr/bin/env python3
"""
tools/plot_preliminary_combined.py — multi-run “intra_inter” edition
with checkpoint-stem/origin tagging, placed right before filenames.

Inputs (tag at the last layer)
──────────────────────────────
results/output/distance/<distance>/<TAG>/seed{seed}.csv
results/output/perlogic/<lr_slug>/<TAG>/{delta.csv,n_total.csv,delta_count.csv}

Outputs (tag at the last layer)
───────────────────────────────
results/figures/<distance>/mSEEDS/mLRS/<TAG>/
    scatter.png
    scatter_binned.png
    scatter_sliding.png
    scatter_acc.png
    correlations.csv               (figure,inter,intra)

Global aggregate (shared):
results/figures/correlations_all.csv
"""
from __future__ import annotations

import argparse
import itertools
import shutil
import uuid
from pathlib import Path
from typing import List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

# ──────────────────────────────────────────────────────────────────────
# Import the "intra_inter" plotting helpers (return (inter, intra) corr)
# ──────────────────────────────────────────────────────────────────────
try:
    # Preferred: package-style import
    from tools.plot_preliminary_intra_inter import (
        plot_basic_scatter,
        plot_rowmean_scatter,
        plot_sliding_scatter,
        plot_accuracy_scatter,
    )
except Exception:
    # Fallback: same-folder import
    from plot_preliminary_intra_inter import (
        plot_basic_scatter,
        plot_rowmean_scatter,
        plot_sliding_scatter,
        plot_accuracy_scatter,
    )


# ═════════════════════════════ utilities ═══════════════════════════════

def parse_list(arg: str | None, cast) -> List:
    """Comma/space–separated CLI list → python list of type `cast`."""
    if arg is None:
        return []
    parts = [p.strip() for seg in arg.split(",") for p in seg.split()]
    return [cast(p) for p in parts if p]


def group_runs(runs: List[Tuple[int, float]], k: int | None) -> List[List[Tuple[int, float]]]:
    """
    Return groups of (seed, lr) runs.
      • k None/0 → one big group (default).
      • k == 1  → one group per run.
      • k  >=2  → all unordered k-combinations of `runs`.
    """
    if not k:
        return [runs]
    if k > len(runs):
        raise ValueError(f"--combine {k} but only {len(runs)} runs supplied")
    # all unordered combinations (so groups can overlap if you iterate all)
    return list(itertools.combinations(runs, k))


def aggregate_mats(mats: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Simple mean aggregation (same shape/index/columns)."""
    stack = np.stack([m.values for m in mats])
    return pd.DataFrame(stack.mean(axis=0), index=mats[0].index, columns=mats[0].columns)


def block_diag_df(mats: List[pd.DataFrame]) -> pd.DataFrame:
    """Block-diagonal concat of square DataFrames; index/columns annotated by block id."""
    if len(mats) == 1:
        return mats[0]
    big = block_diag(*[m.values for m in mats])
    names: List[str] = []
    for k, m in enumerate(mats):
        names.extend([f"{idx}_r{k}" for idx in m.index])
    return pd.DataFrame(big, index=names, columns=names)


def lr_to_slug(lr: float | str) -> str:
    """Canonically slugify an LR value for path use."""
    s = str(lr)
    s = s.replace(".", "p").replace("-", "m")
    return s


# ═════════════════════════════ tagged I/O helpers ═══════════════════════

def resolve_tag(tag: Optional[str], resume: Optional[str], origin_tag: str) -> Optional[str]:
    """
    Priority:
      1) explicit --tag
      2) stem of --resume
      3) origin_tag (default 'origin')
    Returns None if everything empty -> use legacy (untagged) layout.
    """
    if tag:
        return tag
    if resume:
        return Path(resume).stem
    return origin_tag if origin_tag else None


def load_distance(distance: str, seed: int, tag: Optional[str]) -> pd.DataFrame:
    # tag at the last layer under the distance folder
    f = Path(f"results/output/distance/{distance}/seed{seed}.csv") if not tag \
        else Path(f"results/output/distance/{distance}/{tag}/seed{seed}.csv")
    return pd.read_csv(f, index_col=0)


def load_delta(lr: float, tag: Optional[str]) -> pd.DataFrame:
    lr_slug = lr_to_slug(lr)
    f = Path(f"results/output/perlogic/{lr_slug}/delta.csv") if not tag \
        else Path(f"results/output/perlogic/{lr_slug}/{tag}/delta.csv")
    return pd.read_csv(f, index_col=0)


def load_acc_counts(lr: float, tag: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (n_total, delta_count) for accuracy plotting."""
    lr_slug = lr_to_slug(lr)
    root = Path(f"results/output/perlogic/{lr_slug}") if not tag \
        else Path(f"results/output/perlogic/{lr_slug}/{tag}")
    return (
        pd.read_csv(root / "n_total.csv", index_col=0),
        pd.read_csv(root / "delta_count.csv", index_col=0),
    )


# ═════════════════════════════ main ═══════════════════════════════════

def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", required=True, choices=["pot", "edit", "jaccard"])
    # Back-compat singletons
    ap.add_argument("--seed", type=int, help="(back-compat) single seed")
    ap.add_argument("--lr",   type=float, help="(back-compat) single learning rate")
    # Lists override singletons
    ap.add_argument("--seeds", type=str, help="comma/space list, overrides --seed")
    ap.add_argument("--lrs",   type=str, help="comma/space list, overrides --lr")
    # Grouping
    ap.add_argument("--combine", type=int, default=None,
                    help="merge this many (seed,lr) runs into one group (default: all)")
    # Plot params
    ap.add_argument("--win",  type=float, default=0.05)
    ap.add_argument("--step", type=float, default=0.05)
    # Tagging
    ap.add_argument("--tag", type=str, default="", help="Explicit run tag to use in paths")
    ap.add_argument("--resume", type=str, help="Checkpoint path; stem becomes tag if --tag not given")
    ap.add_argument("--origin_tag", type=str, default="origin",
                    help="Tag to use when neither --tag nor --resume is provided")
    args = ap.parse_args()

    # Build tag (or None to use legacy paths)
    TAG = resolve_tag(args.tag, args.resume, args.origin_tag)

    seeds = parse_list(args.seeds, int) or ([args.seed] if args.seed is not None else [])
    lrs   = parse_list(args.lrs,   float) or ([args.lr]   if args.lr   is not None else [])
    if not seeds or not lrs:
        ap.error("Must supply at least one seed and one lr via --seeds/--lrs or --seed/--lr")

    # Build all run pairs, then form groups
    runs   = list(itertools.product(seeds, lrs))      # [(seed, lr), …]
    groups = group_runs(runs, args.combine)           # [[(seed,lr)…], …]

    for gidx, grp in enumerate(groups, 1):
        # Human-friendly tags
        seeds_grp = sorted({s for s, _ in grp})
        lrs_grp   = sorted({lr for _, lr in grp})
        lr_slugs  = [lr_to_slug(lr) for lr in lrs_grp]

        # ── Build block-diag matrices for this group ────────────────────
        dist_cat  = block_diag_df([load_distance(args.distance, s, TAG) for s, _ in grp])
        delta_cat = block_diag_df([load_delta(lr, TAG)                  for _, lr in grp])

        # Accuracy counts also block-diag
        tot_blocks, cor_blocks = [], []
        for _, lr in grp:
            tot, cor = load_acc_counts(lr, TAG)
            tot_blocks.append(tot)
            cor_blocks.append(cor)
        tot_cat = block_diag_df(tot_blocks).round().astype(int)
        cor_cat = block_diag_df(cor_blocks).round().astype(int)

        # Write counts to a temp perlogic root so the accuracy helper can read them
        tmp_root = Path(f"results/output/perlogic/__tmp_{uuid.uuid4().hex}")
        tmp_root.mkdir(parents=True, exist_ok=True)
        (tmp_root / "n_total.csv").write_text(tot_cat.to_csv())
        (tmp_root / "delta_count.csv").write_text(cor_cat.to_csv())

        # Output directory for figures of this group (TAG at the last layer)
        seed_tag = "m" + "-".join(map(str, seeds_grp))
        lr_tag   = "m" + "-".join(lr_slugs)
        out_dir  = Path(f"results/figures/{args.distance}/{seed_tag}/{lr_tag}") if not TAG \
            else Path(f"results/figures/{args.distance}/{seed_tag}/{lr_tag}/{TAG}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Generate four plots; each returns (inter_corr, intra_corr) ──
        stats: List[Tuple[str, float, float]] = []  # (figure, inter, intra)

        ci, cd = plot_basic_scatter(dist_cat, delta_cat, out_dir / "scatter.png")
        stats.append(("scatter", ci, cd))

        ci, cd = plot_rowmean_scatter(dist_cat, delta_cat, out_dir / "scatter_binned.png")
        stats.append(("scatter_binned", ci, cd))

        ci, cd = plot_sliding_scatter(
            dist_cat, delta_cat, out_dir / "scatter_sliding.png",
            win=args.win, step=args.step
        )
        stats.append(("scatter_sliding", ci, cd))

        ci, cd = plot_accuracy_scatter(
            dist_cat, tmp_root, out_dir / "scatter_acc.png",
            win=args.win, step=args.step
        )
        stats.append(("scatter_acc", ci, cd))

        # ── correlations.csv for this group (figure,inter,intra) ────────
        corr_df = pd.DataFrame(stats, columns=["figure", "inter", "intra"]).set_index("figure")
        corr_df.to_csv(out_dir / "correlations.csv")

        # Append to global aggregate (distance,seeds,lrs,figure,inter,intra)
        agg = corr_df.reset_index()
        agg.insert(0, "lrs",   ";".join(map(str, lrs_grp)))
        agg.insert(0, "seeds", ";".join(map(str, seeds_grp)))
        agg.insert(0, "distance", args.distance)

        agg_path = Path(f"results/figures/logs/correlations_{TAG}.csv")
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_path, mode="a", header=not agg_path.exists(), index=False)

        # Cleanup temp counts
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"[✓] group {gidx}: plots → {out_dir}")

if __name__ == "__main__":
    main_cli()