#!/usr/bin/env python3
"""
tools/plot_preliminary.py  ─  unified style, single regression & single r
──────────────────────────────────────────────────────────────────────────
• Four scatter variants (+ optional hist/QQ) exactly as before.
• All points are blue circles; one regression line through them.
• Pairs with interference < 0.1 are dropped (Δ-based plots only).
• Each plot writes ONE Pearson r (over all kept points).
  Per-combo CSV header:  figure,corr
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress

# ──────────────── Matplotlib global style ───────────────────────────────
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "axes.facecolor": "#EEF0F2",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "gray",
    "grid.alpha": 0.5,
})
rcParams["font.size"] = 16

LABEL0 = {0, 1, 3, 4, 8}                 # edit if you change the 10-logic split
BLUE   = (76/255, 114/255, 176/255)

# ══════════════════════════════ helpers ════════════════════════════════
def _logic_labels(n: int) -> np.ndarray:
    return np.array([(i % 10) in LABEL0 for i in range(n)], dtype=bool)


def _style_axes(ax, xlab: str, ylab: str):
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_facecolor("#EEF0F2")
    ax.grid(True, linestyle="--", color="gray", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()


def _add_corr_line(ax, x: np.ndarray, y: np.ndarray):
    if len(x) > 1 and x.std() and y.std():
        slope, intercept, *_ = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 120)
        ax.plot(xs, slope * xs + intercept,
                linestyle="--", linewidth=2, color="black", alpha=0.65)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and x.std() and y.std() else float("nan")


def _filter_pairs(x: np.ndarray, y: np.ndarray, mask: np.ndarray):
    sel = mask & (y >= 0.1)
    return x[sel], y[sel]


def _plot(ax, x: np.ndarray, y: np.ndarray):
    ax.scatter(x, y, s=320, marker="o",
               facecolor=BLUE, edgecolor="black",
               linewidth=1.4, alpha=.85)

# ══════════════════════════════ scatter variants ═══════════════════════
def plot_basic_scatter(dist: pd.DataFrame, delta: pd.DataFrame, out: Path,
                       xlab="Distance", ylab="Interference") -> float:
    out.parent.mkdir(parents=True, exist_ok=True)
    x_all, y_all = dist.values.flatten(), delta.values.flatten()
    n = len(dist)
    mask = (x_all != 0) & (y_all >= 0.1)          # filter once

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot(ax, x_all[mask], y_all[mask])
    _add_corr_line(ax, x_all[mask], y_all[mask])
    _style_axes(ax, xlab, ylab)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return _corr(x_all[mask], y_all[mask])


def plot_rowmean_scatter(dist: pd.DataFrame, delta: pd.DataFrame, out: Path,
                         xlab="Distance", ylab="Interference") -> float:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(dist)
    off_diag = ~np.eye(n, dtype=bool)

    xs, ys = [], []
    for i in range(n):
        js = off_diag[i] & (delta.values[i] >= 0.1) & (dist.values[i] != 0)
        if js.any():
            xs.append(dist.values[i][js].mean())
            ys.append(delta.values[i][js].mean())
    diag_mask = delta.values.diagonal() >= 0.1
    xs.extend(dist.values.diagonal()[diag_mask])
    ys.extend(delta.values.diagonal()[diag_mask])

    xs, ys = np.asarray(xs), np.asarray(ys)
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot(ax, xs, ys)
    _add_corr_line(ax, xs, ys)
    _style_axes(ax, xlab, ylab)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return _corr(xs, ys)


def plot_sliding_scatter(dist: pd.DataFrame, delta: pd.DataFrame, out: Path,
                         win=.10, step=.10, xlab="Distance", ylab="Interference") -> float:
    out.parent.mkdir(parents=True, exist_ok=True)
    x_all, y_all = dist.values.flatten(), delta.values.flatten()
    valid = (x_all != 0) & (y_all >= 0.1)

    xs, ys = [], []
    lo, hi = x_all[valid].min(), x_all[valid].max()
    for left in np.arange(lo, hi - win + 1e-9, step):
        sel = valid & (x_all >= left) & (x_all < left + win)
        if sel.any():
            xs.append(x_all[sel].mean())
            ys.append(y_all[sel].mean())
    xs, ys = np.asarray(xs), np.asarray(ys)

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot(ax, xs, ys)
    _add_corr_line(ax, xs, ys)
    _style_axes(ax, xlab, ylab)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return _corr(xs, ys)


def plot_accuracy_scatter(dist: pd.DataFrame, perlogic_root: Path, out: Path,
                          win=.10, step=.10, xlab="Distance") -> float:
    from warnings import warn
    out.parent.mkdir(parents=True, exist_ok=True)

    p_tot, p_cor = perlogic_root / "n_total.csv", perlogic_root / "delta_count.csv"
    if not (p_tot.exists() and p_cor.exists()):
        warn("missing accuracy files – skipping accuracy plot")
        return float("nan")

    mx    = dist
    m_tot = pd.read_csv(p_tot, index_col=0).loc[mx.index, mx.columns]
    m_cor = pd.read_csv(p_cor, index_col=0).loc[mx.index, mx.columns]
    m_acc = m_cor.where(m_tot != 0) / m_tot

    x_all, a_all, t_all = mx.values.flatten(), m_acc.values.flatten(), m_tot.values.flatten()
    mask  = ~np.isnan(a_all)

    xs, ys = [], []
    lo, hi = x_all[mask].min(), x_all[mask].max()
    for left in np.arange(lo, hi - win + 1e-9, step):
        sel = mask & (x_all >= left) & (x_all < left + win)
        if sel.any():
            acc = (a_all[sel] * t_all[sel]).sum() / t_all[sel].sum()
            xs.append(x_all[sel].mean())
            ys.append(acc)
    xs, ys = np.asarray(xs), np.asarray(ys)

    fig, ax = plt.subplots(figsize=(6, 5))
    _plot(ax, xs, ys)
    _add_corr_line(ax, xs, ys)
    _style_axes(ax, xlab, "Accuracy")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return _corr(xs, ys)

# ══════════════════════════════ CLI / entrypoint ═══════════════════════
def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--distance", required=True, choices=["pot", "edit", "jaccard"])
    p.add_argument("--seed",     type=int,  required=True)
    p.add_argument("--lr",       type=float,required=True)
    p.add_argument("--win",      type=float,default=.05)
    p.add_argument("--step",     type=float,default=.05)
    p.add_argument("--plot_dist",action="store_true")
    return p


def main_cli():
    args = _parser().parse_args()
    lr_slug = str(args.lr).replace(".", "p").replace("-", "m")

    dist_csv  = Path(f"results/output/distance/{args.distance}/seed{args.seed}.csv")
    delta_csv = Path(f"results/output/perlogic/{lr_slug}/delta.csv")
    dist_mat  = pd.read_csv(dist_csv,  index_col=0)
    delta_mat = pd.read_csv(delta_csv, index_col=0).loc[dist_mat.index, dist_mat.columns]

    out_dir = Path(f"results/figures/{args.distance}/seed{args.seed}/lr{lr_slug}")
    stats: List[Tuple[str, float]] = []         # figure, corr

    stats.append(("scatter",
                  plot_basic_scatter(dist_mat, delta_mat, out_dir / "scatter.png")))
    stats.append(("scatter_binned",
                  plot_rowmean_scatter(dist_mat, delta_mat, out_dir / "scatter_binned.png")))
    stats.append(("scatter_sliding",
                  plot_sliding_scatter(dist_mat, delta_mat, out_dir / "scatter_sliding.png",
                                       win=args.win, step=args.step)))
    stats.append(("scatter_acc",
                  plot_accuracy_scatter(dist_mat,
                                        Path(f"results/output/perlogic/{lr_slug}"),
                                        out_dir / "scatter_acc.png",
                                        win=args.win, step=args.step)))

    combo_df = pd.DataFrame(stats, columns=["figure", "corr"])
    combo_path = out_dir / "correlations.csv"
    combo_df.to_csv(combo_path, index=False)
    print(f"[info] saved → {combo_path}")

    agg_df = combo_df.copy()
    agg_df.insert(0, "lr", args.lr)
    agg_df.insert(0, "seed", args.seed)
    agg_df.insert(0, "distance", args.distance)

    agg_path = Path("results/figures/correlations_all.csv")
    agg_df.to_csv(agg_path, mode="a", header=not agg_path.exists(), index=False)
    print(f"[info] appended → {agg_path}")

    if args.plot_dist:
        from tools.plot_preliminary import plot_dist_vs_delta_hist, plot_qq_dist_delta
        plot_dist_vs_delta_hist(dist_mat, delta_mat, out_dir / "hist_distance_delta.png")
        plot_qq_dist_delta(dist_mat, delta_mat, out_dir / "qq_distance_delta.png")


if __name__ == "__main__":
    main_cli()