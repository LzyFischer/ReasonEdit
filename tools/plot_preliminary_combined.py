#!/usr/bin/env python3
"""
tools/plot_preliminary.py  —  multi-run edition, unified-corr
with checkpoint-stem/origin tagging support.

Inputs (tagged):
  results/output/distance/<TAG>/<distance>/seed{seed}.csv
  results/output/perlogic/<TAG>/<lr_slug>/{delta.csv,n_total.csv,delta_count.csv}

Outputs (tagged):
  results/figures/<distance>/<TAG>/mSEEDS/mLRS/
    scatter.png, scatter_binned.png, scatter_sliding.png, scatter_acc.png, correlations.csv
Global aggregate (per TAG):
  results/figures/<TAG>/correlations_all.csv
"""
from __future__ import annotations
import argparse, itertools, uuid, shutil
from pathlib import Path
from typing import List, Tuple, Sequence, Optional
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

# updated plotting helpers (single-corr)
from tools.plot_preliminary import (
    plot_basic_scatter,
    plot_rowmean_scatter,
    plot_sliding_scatter,
    plot_accuracy_scatter,
)

# ═════════════════════════════ helpers ════════════════════════════════
def parse_list(arg: str | None, cast) -> List:
    """Comma/space–separated CLI list → python list of `cast`."""
    if arg is None:
        return []
    parts = [p.strip() for seg in arg.split(",") for p in seg.split()]
    return [cast(p) for p in parts if p]

def group_runs(runs: List[Tuple[int, float]], k: int | None) -> List[List[Tuple[int, float]]]:
    """
    Return *all* unordered k-size combinations of `runs`.
      • k None/0 → one big group (default).
      • k == 1  → one group per run.
    """
    if not k:
        return [runs]
    if k > len(runs):
        raise ValueError(f"--combine {k} but only {len(runs)} runs supplied")
    return list(combinations(runs, k))

def aggregate_mats(mats: Sequence[pd.DataFrame]) -> pd.DataFrame:
    stack = np.stack([m.values for m in mats])
    return pd.DataFrame(stack.mean(axis=0), index=mats[0].index, columns=mats[0].columns)

def lr_slug(lr: float | str) -> str:
    return str(lr).replace(".", "p").replace("-", "m")

def resolve_tag(tag: Optional[str], resume: Optional[str], origin_tag: str) -> Optional[str]:
    """
    Priority: explicit --tag → stem(--resume) → origin_tag (default 'origin').
    Returns None to use legacy (untagged) layout if everything empty.
    """
    if tag:
        return tag
    if resume:
        return Path(resume).stem
    return origin_tag if origin_tag else None

# ───────────── tagged I/O helpers (fall back to legacy if TAG is None) ─────────────
def load_distance(distance: str, seed: int, TAG: Optional[str]) -> pd.DataFrame:
    f = (Path(f"results/output/distance/{distance}/{TAG}/seed{seed}.csv")
         if TAG else Path(f"results/output/distance/{distance}/seed{seed}.csv"))
    return pd.read_csv(f, index_col=0)

def load_delta(lr: float, TAG: Optional[str]) -> pd.DataFrame:
    s = lr_slug(lr)
    f = (Path(f"results/output/perlogic/{s}/{TAG}/delta.csv")
         if TAG else Path(f"results/output/perlogic/{s}/delta.csv"))
    return pd.read_csv(f, index_col=0)

def load_acc_counts(lr: float, TAG: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    s = lr_slug(lr)
    root = (Path(f"results/output/perlogic/{s}/{TAG}")
            if TAG else Path(f"results/output/perlogic/{s}"))
    return (pd.read_csv(root / "n_total.csv", index_col=0),
            pd.read_csv(root / "delta_count.csv", index_col=0))

def block_diag_df(mats: List[pd.DataFrame]) -> pd.DataFrame:
    if len(mats) == 1:
        return mats[0]
    big = block_diag(*[m.values for m in mats])
    names = []
    for k, m in enumerate(mats):
        names.extend([f"{idx}_r{k}" for idx in m.index])
    return pd.DataFrame(big, index=names, columns=names)

# ═════════════════════════════ main ═══════════════════════════════════
def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", required=True, choices=["pot", "edit", "jaccard"])
    ap.add_argument("--seed", type=int, help="(back-compat) single seed")
    ap.add_argument("--lr",   type=float, help="(back-compat) single learning rate")
    ap.add_argument("--seeds", type=str, help="comma/space list, overrides --seed")
    ap.add_argument("--lrs",   type=str, help="comma/space list, overrides --lr")
    ap.add_argument("--combine", type=int, default=None,
                    help="merge this many (seed,lr) runs into one group")
    ap.add_argument("--win",  type=float, default=.05)
    ap.add_argument("--step", type=float, default=.05)
    # NEW: tagging
    ap.add_argument("--tag", type=str, default="", help="Explicit run tag to use in paths")
    ap.add_argument("--resume", type=str, help="Checkpoint path; stem becomes tag if --tag not given")
    ap.add_argument("--origin_tag", type=str, default="origin",
                    help="Tag to use when neither --tag nor --resume is provided")
    args = ap.parse_args()

    TAG = resolve_tag(args.tag, args.resume, args.origin_tag)

    seeds = parse_list(args.seeds, int)  or ([args.seed] if args.seed is not None else [])
    lrs   = parse_list(args.lrs,   float) or ([args.lr]   if args.lr   is not None else [])
    if not seeds or not lrs:
        ap.error("Must supply at least one seed and one lr via --seeds/--lrs or --seed/--lr")

    runs   = list(itertools.product(seeds, lrs))   # [(seed, lr), …]
    groups = group_runs(runs, args.combine)        # [[(seed,lr)…], …]

    for gidx, grp in enumerate(groups, 1):
        seeds_grp = sorted({s for s, _ in grp})
        lrs_grp   = sorted({lr for _, lr in grp})
        lr_slugs  = [lr_slug(lr) for lr in lrs_grp]

        # ── build block-diag matrices ───────────────────────────────────
        dist_cat  = block_diag_df([load_distance(args.distance, s, TAG) for s, _ in grp])
        delta_cat = block_diag_df([load_delta(lr, TAG)                  for _, lr in grp])

        tot_blocks, cor_blocks = [], []
        for _, lr in grp:
            tot, cor = load_acc_counts(lr, TAG)
            tot_blocks.append(tot)
            cor_blocks.append(cor)
        tot_cat = block_diag_df(tot_blocks).round().astype(int)
        cor_cat = block_diag_df(cor_blocks).round().astype(int)

        # write to a temp dir so accuracy helper can read counts
        tmp_root = Path(f"results/output/perlogic/__tmp_{uuid.uuid4().hex}")
        tmp_root.mkdir(parents=True, exist_ok=True)
        tot_cat.to_csv(tmp_root / "n_total.csv")
        cor_cat.to_csv(tmp_root / "delta_count.csv")

        # figure output dir (tagged or legacy)
        seed_tag = "m" + "-".join(map(str, seeds_grp))
        lr_tag   = "m" + "-".join(lr_slugs)
        if TAG:
            out_dir = Path(f"results/figures/{args.distance}/{TAG}/{seed_tag}/{lr_tag}")
            agg_path = Path(f"results/figures/{TAG}/correlations_all.csv")
        else:
            out_dir = Path(f"results/figures/{args.distance}/{seed_tag}/{lr_tag}")
            agg_path = Path("results/figures/correlations_all.csv")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── generate four plots ─────────────────────────────────────────
        stats: List[Tuple[str, float]] = []
        stats.append(("scatter",
            plot_basic_scatter(dist_cat, delta_cat, out_dir / "scatter.png")))
        stats.append(("scatter_binned",
            plot_rowmean_scatter(dist_cat, delta_cat, out_dir / "scatter_binned.png")))
        stats.append(("scatter_sliding",
            plot_sliding_scatter(dist_cat, delta_cat,
                                 out_dir / "scatter_sliding.png",
                                 win=args.win, step=args.step)))
        stats.append(("scatter_acc",
            plot_accuracy_scatter(dist_cat, tmp_root,
                                  out_dir / "scatter_acc.png",
                                  win=args.win, step=args.step)))

        # correlations.csv for this group
        corr_df = pd.DataFrame(stats, columns=["figure", "corr"]).set_index("figure")
        corr_df.to_csv(out_dir / "correlations.csv")

        # append to global aggregate
        agg = corr_df.reset_index()
        agg.insert(0, "lrs",   ";".join(map(str, lrs_grp)))
        agg.insert(0, "seeds", ";".join(map(str, seeds_grp)))
        agg.insert(0, "distance", args.distance)

        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_path, mode="a", header=not agg_path.exists(), index=False)

        # cleanup
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"[✓] group {gidx}: plots → {out_dir}")

if __name__ == "__main__":
    main_cli()