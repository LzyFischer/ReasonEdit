#!/usr/bin/env python3
"""
tools/plot_preliminary.py  —  multi-run edition
────────────────────────────────────────────────────────────────────────────
• Accepts *lists* of --seeds and --lrs and an optional --combine SIZE.
• Averages distance & Δ-accuracy matrices over each SIZE-sized chunk
  of runs, then draws the four standard figures **per chunk**.
• Still writes per-combo correlations.csv *and* appends to the
  global correlations_all.csv.

Run example
└─ python -m tools.plot_preliminary \
       --distance pot \
       --seeds 10,12,13,14,15,17,18,19 \
       --lrs 1e-4,1.5e-4 \
       --combine 6
"""
from __future__ import annotations
import argparse, itertools, uuid, shutil
from pathlib import Path
from typing import List, Tuple, Sequence, Dict
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from itertools import combinations
from tools.plot_preliminary import plot_basic_scatter, plot_rowmean_scatter, plot_accuracy_scatter, plot_sliding_scatter

# ── original helper-functions (unchanged) ────────────────────────────────
# … <keep every existing helper here> …

# ─────────────────────────────────────────────────────────────────────────
def parse_list(arg: str, cast) -> List:
    """Convert a comma/space-separated CLI list into python list of `cast`."""
    if arg is None:
        return []
    parts = [p.strip() for seg in arg.split(",") for p in seg.split()]
    return [cast(p) for p in parts if p != ""]

def group_runs(runs: List[Tuple[int, float]], k: int | None) -> List[List[Tuple[int, float]]]:
    """
    Enumerate **all** unordered k-size combinations of `runs`.
      • If k is None/0 → one big group with every run (original behaviour).
      • If k==1       → one group per run.
      • If k>len(runs)→ error.
    """
    if k is None or k == 0:
        return [runs]
    if k > len(runs):
        raise ValueError(f"--combine {k} but only {len(runs)} runs supplied")
    return list(combinations(runs, k))

# ─────────────────────────────────────────────────────────────────────────
def aggregate_mats(mats: Sequence[pd.DataFrame]) -> pd.DataFrame:
    stack = np.stack([m.values for m in mats])
    mean = stack.mean(axis=0)
    return pd.DataFrame(mean, index=mats[0].index, columns=mats[0].columns)

# ─────────────────────────────────────────────────────────────────────────
def load_distance(distance: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(Path(f"results/output/distance/{distance}/seed{seed}.csv"),
                       index_col=0)

def load_delta(lr: float) -> pd.DataFrame:
    lr_slug = str(lr).replace(".", "p").replace("-", "m")
    return pd.read_csv(Path(f"results/output/perlogic/{lr_slug}/delta.csv"),
                       index_col=0)

def load_acc_counts(lr: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lr_slug = str(lr).replace(".", "p").replace("-", "m")
    root = Path(f"results/output/perlogic/{lr_slug}")
    tot = pd.read_csv(root / "n_total.csv",    index_col=0)
    cor = pd.read_csv(root / "delta_count.csv", index_col=0)
    return tot, cor

def block_diag_df(mats: List[pd.DataFrame]) -> pd.DataFrame:
    if len(mats) == 1:
        return mats[0]
    big = block_diag(*[m.values for m in mats])
    # 给行列起唯一名字，方便 debug（逻辑标签只看索引 mod 10）
    names = []
    for k, m in enumerate(mats):
        names.extend([f"{idx}_r{k}" for idx in m.index])
    return pd.DataFrame(big, index=names, columns=names)

# ─────────────────────────────────────────────────────────────────────────
def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", required=True, choices=["pot", "edit", "jaccard"])
    ap.add_argument("--seed", type=int, help="(back-compat) single seed")
    ap.add_argument("--lr",   type=float, help="(back-compat) single learning rate")

    ap.add_argument("--seeds", type=str, help="comma/space list, overrides --seed")
    ap.add_argument("--lrs",   type=str, help="comma/space list, overrides --lr")
    ap.add_argument("--combine", type=int, default=None,
                    help="how many (seed,lr) runs to merge into one group")

    # original extra flags (win, step …) stay unchanged
    ap.add_argument("--win",  type=float, default=.01)
    ap.add_argument("--step", type=float, default=.01)
    args = ap.parse_args()

    seeds = parse_list(args.seeds, int)  or ([args.seed] if args.seed is not None else [])
    lrs   = parse_list(args.lrs, float) or ([args.lr]   if args.lr   is not None else [])

    if not seeds or not lrs:
        ap.error("Must supply at least one seed and one lr via --seeds/--lrs or --seed/--lr")

    runs     = list(itertools.product(seeds, lrs))           # [(seed, lr), …]
    groups   = group_runs(runs, args.combine)                # [[(seed,lr)…], …]

    for idx, grp in enumerate(groups, 1):
        seeds_in_grp = sorted({s for s, _ in grp})
        lrs_in_grp   = sorted({lr for _, lr in grp})
        lr_slugs     = [str(lr).replace(".", "p").replace("-", "m") for lr in lrs_in_grp]

        # ── aggregate matrices ──────────────────────────────────────────
        dist_mats  = [load_distance(args.distance, s) for s, _ in grp]
        delta_mats = [load_delta(lr)                  for _, lr in grp]

        dist_cat   = block_diag_df(dist_mats)
        delta_cat  = block_diag_df(delta_mats)

        # ── aggregate accuracy counts (n_total / delta_count) ───────────
        tot_list, cor_list = [], []
        for _, lr in grp:
            t,c = load_acc_counts(lr)
            tot_list.append(t)
            cor_list.append(c)
        # tot_sum = aggregate_mats(tot_list).round().astype(int)
        # cor_sum = aggregate_mats(cor_list).round().astype(int)
        tot_sum = block_diag_df(tot_list).round().astype(int)
        cor_sum = block_diag_df(cor_list).round().astype(int)

        # dump to a temp folder so the existing helper can read them
        tmp_root = Path(f"results/output/perlogic/__tmp_{uuid.uuid4().hex}")
        tmp_root.mkdir(parents=True, exist_ok=True)
        (tmp_root / "n_total.csv").write_text(tot_sum.to_csv())
        (tmp_root / "delta_count.csv").write_text(cor_sum.to_csv())

        # ── figure output directory ─────────────────────────────────────
        seed_tag = "m" + "-".join(map(str, seeds_in_grp))      # m = merged
        lr_tag   = "m" + "-".join(lr_slugs)
        out_dir  = Path(f"results/figures/{args.distance}/{seed_tag}/{lr_tag}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── generate the four plots (functions unchanged) ───────────────
        stats: List[Tuple[str,float,float]] = []

        ci, cd = plot_basic_scatter(dist_cat, delta_cat, out_dir / "scatter.png")
        stats.append(("scatter", ci, cd))

        ci, cd = plot_rowmean_scatter(dist_cat, delta_cat,
                                      out_dir / "scatter_binned.png")
        stats.append(("scatter_binned", ci, cd))

        ci, cd = plot_sliding_scatter(dist_cat, delta_cat,
                                      out_dir / "scatter_sliding.png",
                                      win=args.win, step=args.step)
        stats.append(("scatter_sliding", ci, cd))

        ci, cd = plot_accuracy_scatter(dist_cat, tmp_root,
                                       out_dir / "scatter_acc.png",
                                       win=args.win, step=args.step)
        stats.append(("scatter_acc", ci, cd))

        # ── correlations.csv for this group ─────────────────────────────
        corr_df = (pd.DataFrame(stats, columns=["figure","inter","intra"])
                     .set_index("figure"))
        corr_df.to_csv(out_dir / "correlations.csv")

        # append to global aggregate
        agg = corr_df.reset_index()
        agg.insert(0, "lrs",  ";".join(map(str, lrs_in_grp)))
        agg.insert(0, "seeds",";".join(map(str, seeds_in_grp)))
        agg.insert(0, "distance", args.distance)
        agg_path = Path("results/figures/correlations_all.csv")
        agg.to_csv(agg_path, mode="a", header=not agg_path.exists(), index=False)

        # clean temp folder
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"[info] group {idx}: plots → {out_dir}")

if __name__ == "__main__":
    main_cli()