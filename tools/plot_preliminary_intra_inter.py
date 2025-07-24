#!/usr/bin/env python3
"""
Enhanced tools/plot_preliminary.py
────────────────────────────────────────────────────────────────────────────
• Produces four scatter plots as before.
• Computes *negative* Pearson r for Inter‑Pattern & Intra‑Pattern points of each
  figure, pivots the result to columns **inter**/**intra** (rows = figure).
• Writes the per‑combo correlations to:
    results/figures/{distance}/seed{seed}/lr{lr_slug}/correlations.csv
  with header: ``figure,inter,intra``.
• Appends the same rows—augmented with distance/seed/lr metadata—to an
  aggregate file:
    results/figures/correlations_all.csv
  (columns: distance,seed,lr,figure,inter,intra).
Run example
└─ ``python -m tools.plot_preliminary --distance pot --seed 0 --lr 1e-4``
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
from scipy.stats import linregress
from scipy import stats
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ──────────────── Matplotlib global style  ───────────────────────────────
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

# 10‑logic fixed grouping; edit if necessary
LABEL0 = {0, 1, 3, 4, 8}  # {logic_000,001,003,004,008} ⇒ group‑0
PALETTE = [
    (255/255, 190/255, 122/255),  # peach (intra)
    (250/255, 127/255, 111/255),  # salmon (unused)
    (130/255, 176/255, 210/255),  # sky‑blue (inter)
    (142/255, 207/255, 201/255),  # teal    (unused)
]

# ══════════════════════════════ helpers ════════════════════════════════
def plot_dist_vs_delta_hist(dist: pd.DataFrame,
                            delta: pd.DataFrame,
                            out: Path,
                            bins=40) -> None:
    """
    把 distance 与 delta 打平成一维后：
      • 绘制重叠直方图（共享 bins）与 KDE 曲线
      • 同时输出 Kolmogorov–Smirnov p-value 作为参考
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    d1 = dist.values.flatten()
    d2 = delta.values.flatten()

    # 可选：去掉 0（或 NaN / inf）
    m = (np.isfinite(d1)) & (d1 != 0)
    n = (np.isfinite(d2)) & (d2 != 0)
    d1, d2 = d1[m], d2[n]

    # 统计检验：KS
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(d1, d2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d1, bins=bins, density=True, alpha=.45,
            label=f"distance (n={len(d1)})")
    ax.hist(d2, bins=bins, density=True, alpha=.45,
            label=f"delta (n={len(d2)})")

    # KDE
    from scipy.stats import gaussian_kde
    xs = np.linspace(min(d1.min(), d2.min()),
                     max(d1.max(), d2.max()), 400)
    ax.plot(xs, gaussian_kde(d1)(xs), lw=2)
    ax.plot(xs, gaussian_kde(d2)(xs), lw=2)

    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title(f"distance vs delta  (KS-p = {ks_p:.2e})")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=.5)
    plt.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_qq_dist_delta(dist: pd.DataFrame,
                       delta: pd.DataFrame,
                       out: Path) -> None:
    """
    双样本 QQ-plot：distance 作为横轴，delta 作为纵轴
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    x = dist.values.flatten()
    y = delta.values.flatten()

    mask = (np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0))
    x, y = np.sort(x[mask]), np.sort(y[mask])
    k = min(len(x), len(y))           # 取共同长度
    x, y = x[:k], y[:k]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=25, color="#4C72B0", alpha=.8)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, "--k", lw=1)    # y=x 参考线
    ax.set_xlabel("distance quantiles")
    ax.set_ylabel("delta quantiles")
    ax.set_title("QQ-plot  (distance  vs  delta)")
    ax.grid(True, linestyle=":", alpha=.5)
    plt.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _logic_labels(n: int) -> np.ndarray:
    return np.array([0 if (i % 10) in LABEL0 else 1 for i in range(n)],
                    dtype=int)


def _style_axes(ax, xlab: str, ylab: str):
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_facecolor("#EEF0F2")
    ax.grid(True, linestyle="--", color="gray", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, markerscale=0.8)
    plt.tight_layout()


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return the *negative* Pearson correlation coefficient (−r).
    NaN if undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

# ══════════════════════════════ plotting helpers ═══════════════════════
def _add_corr_line(ax, x, y):
    """Add regression line if valid."""
    if len(x) > 1 and not np.isclose(np.std(x), 0) and not np.isclose(np.std(y), 0):
        slope, intercept, _, _, _ = linregress(x, y)
        x_vals = np.linspace(min(x), max(x), 100)
        y_vals = slope * x_vals + intercept
        ax.plot(x_vals, y_vals, linestyle="--", linewidth=2, color="black", alpha=0.6)

def plot_basic_scatter(dist: pd.DataFrame, delta: pd.DataFrame, out: Path,
                       dist_name="Distance", delta_name="Interference") -> Tuple[float, float]:
    """Point‑wise scatter. Returns (corr_inter, corr_intra)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    x, y = dist.values.flatten(), delta.values.flatten()
    n = len(dist)

    off_diag = ~np.eye(n, dtype=bool)
    diff_lbl = _logic_labels(n)[:, None] != _logic_labels(n)[None, :]
    mask_inter = (off_diag & diff_lbl).flatten() & ((x != 0) & (y != 0))
    mask_intra = (~off_diag).flatten()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x[mask_inter], y[mask_inter], s=400, marker="o",
               facecolor=PALETTE[2], edgecolor="black", linewidth=1.8,
               alpha=.8, label="Inter‑Pattern")
    ax.scatter(x[mask_intra], y[mask_intra], s=400, marker="s",
               facecolor=PALETTE[0], edgecolor="black", linewidth=1.8,
               alpha=.8, label="Intra‑Pattern")
    rows, cols = np.divmod(np.arange(n * n), n)          # pre-compute all indices
    for idx in np.where(mask_inter | mask_intra)[0]:     # only annotate shown points
        r, c = rows[idx], cols[idx]
        ax.text(x[idx], y[idx], f"{r},{c}",
                fontsize=9, ha="center", va="center", alpha=.9)
    
    _add_corr_line(ax, x[mask_inter], y[mask_inter])
    _add_corr_line(ax, x[mask_intra], y[mask_intra])

    _style_axes(ax, dist_name, delta_name)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return _corr(x[mask_inter], y[mask_inter]), _corr(x[mask_intra], y[mask_intra])


def plot_rowmean_scatter(dist: pd.DataFrame, delta: pd.DataFrame, out: Path,
                         dist_name="Distance", delta_name="Interference") -> Tuple[float, float]:
    """Row‑mean scatter."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(dist)
    lbl = _logic_labels(n)
    off_diag = ~np.eye(n, dtype=bool)
    diff_lbl = lbl[:, None] != lbl[None, :]

    vals_x, vals_y = [], []
    for i in range(n):
        js = np.where(off_diag[i] & diff_lbl[i] & (dist.values[i] != 0) & (delta.values[i] != 0))[0]
        if js.size:
            vals_x.append(dist.iloc[i, js].mean())
            vals_y.append(delta.iloc[i, js].mean())

    diag_x, diag_y = np.diag(dist), np.diag(delta)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(vals_x, vals_y, s=400, marker="o", facecolor=PALETTE[2],
               edgecolor="black", linewidth=1.8, alpha=.8,
               label="Inter‑Pattern (row‑mean)")
    ax.scatter(diag_x, diag_y, s=400, marker="s", facecolor=PALETTE[0],
               edgecolor="black", linewidth=1.8, alpha=.8, label="Intra‑Pattern")
    _add_corr_line(ax, vals_x, vals_y)
    _add_corr_line(ax, diag_x, diag_y)
    
    _style_axes(ax, dist_name, delta_name)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return _corr(vals_x, vals_y), _corr(diag_x, diag_y)


def plot_sliding_scatter(dist: pd.DataFrame, delta: pd.DataFrame, out: Path,
                         win=.10, step=.10,
                         dist_name="Distance", delta_name="Interference") -> Tuple[float, float]:
    """Sliding‑window Δ vs distance scatter."""
    out.parent.mkdir(parents=True, exist_ok=True)
    x, y = dist.values.flatten(), delta.values.flatten()
    n = len(dist)
    labels = _logic_labels(n)

    off_diag = ~np.eye(n, dtype=bool)
    diff_lbl = labels[:, None] != labels[None, :]
    nonzero = (x != 0) & (y != 0)
    m_inter = (off_diag & diff_lbl).flatten() & nonzero
    m_intra = (~off_diag).flatten()

    def slide(xx, yy, mask):
        if not mask.any():
            return np.array([]), np.array([])
        xx, yy = xx[mask], yy[mask]
        xs, ys = [], []
        lo, hi = xx.min(), xx.max()
        for left in np.arange(lo, hi - win + 1e-9, step):
            sel = (xx >= left) & (xx < left + win)
            if sel.any():
                xs.append(xx[sel].mean())
                ys.append(yy[sel].mean())
        return np.asarray(xs), np.asarray(ys)

    xi, yi = slide(x, y, m_inter)
    xd, yd = slide(x, y, m_intra)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xi, yi, s=400, marker="o", facecolor=PALETTE[2],
               edgecolor="black", linewidth=1.8, alpha=.8, label="Inter‑Pattern")
    ax.scatter(xd, yd, s=400, marker="s", facecolor=PALETTE[0],
               edgecolor="black", linewidth=1.8, alpha=.8, label="Intra‑Pattern")
    logic_names = list(dist.index)       # or None if you prefer pure indices

    _add_corr_line(ax, xi, yi)
    _add_corr_line(ax, xd, yd)

    _style_axes(ax, dist_name, delta_name)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return _corr(xi, yi), _corr(xd, yd)


def plot_accuracy_scatter(dist: pd.DataFrame, perlogic_root: Path, out: Path,
                          win=.10, step=.10, dist_name="Distance") -> Tuple[float, float]:
    """Sliding‑window *accuracy* vs distance scatter."""
    out.parent.mkdir(parents=True, exist_ok=True)

    path_tot, path_cor = perlogic_root / "n_total.csv", perlogic_root / "delta_count.csv"
    if not (path_tot.exists() and path_cor.exists()):
        print(f"[warn] {path_tot.name} or {path_cor.name} missing → skip accuracy plot")
        return float("nan"), float("nan")

    mx = dist
    m_tot = pd.read_csv(path_tot, index_col=0).loc[mx.index, mx.columns]
    m_cor = pd.read_csv(path_cor, index_col=0).loc[mx.index, mx.columns]
    m_acc = m_cor.where(m_tot != 0) / m_tot

    x, acc, tot = mx.values.flatten(), m_acc.values.flatten(), m_tot.values.flatten()
    n = len(mx)
    labels = _logic_labels(n)

    off_diag = ~np.eye(n, dtype=bool)
    diff_lbl = labels[:, None] != labels[None, :]
    nonnan   = ~np.isnan(acc)
    m_inter  = (off_diag & diff_lbl).flatten() & (x != 0) & nonnan
    m_intra  = (~off_diag).flatten() & nonnan

    def window(xx, a, t, mask):
        if not mask.any():
            return np.array([]), np.array([])
        xx, a, t = xx[mask], a[mask], t[mask]
        xs, ys = [], []
        lo, hi = xx.min(), xx.max()
        for left in np.arange(lo, hi - win + 1e-9, step):
            sel = (xx >= left) & (xx < left + win)
            if sel.any():
                numer, denom = (a[sel] * t[sel]).sum(), t[sel].sum()
                xs.append(xx[sel].mean())
                ys.append(numer / denom)
        return np.asarray(xs), np.asarray(ys)

    xi, yi = window(x, acc, tot, m_inter)
    xd, yd = window(x, acc, tot, m_intra)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xi, yi, s=400, marker="o", facecolor=PALETTE[2],
               edgecolor="black", linewidth=1.8, alpha=.8, label="Inter‑Pattern")
    ax.scatter(xd, yd, s=400, marker="s", facecolor=PALETTE[0],
               edgecolor="black", linewidth=1.8, alpha=.8, label="Intra‑Pattern")
    _add_corr_line(ax, xi, yi)
    _add_corr_line(ax, xd, yd)

    _style_axes(ax, dist_name, "Interference")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return _corr(xi, yi), _corr(xd, yd)

# ══════════════════════════════ CLI / entrypoint ═══════════════════════

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--distance", required=True, choices=["pot", "edit", "jaccard"],
                   help="Distance metric name (folder under results/output/distance)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--win", type=float, default=.05)
    p.add_argument("--step", type=float, default=.05)
    p.add_argument("--plot_dist", action="store_true")
    return p


def main_cli():
    args = _parser().parse_args()
    lr_slug = str(args.lr).replace(".", "p").replace("-", "m")

    # ── load matrices ───────────────────────────────────────────────────
    dist_csv = Path(f"results/output/distance/{args.distance}/seed{args.seed}.csv")
    delta_csv = Path(f"results/output/perlogic/{lr_slug}/delta.csv")
    dist_mat = pd.read_csv(dist_csv, index_col=0)
    delta_mat = pd.read_csv(delta_csv, index_col=0).loc[dist_mat.index, dist_mat.columns]

    out_dir = Path(f"results/figures/{args.distance}/seed{args.seed}/lr{lr_slug}")

    stats: List[Tuple[str, float, float]] = []  # figure, inter, intra

    # ① basic
    ci, cd = plot_basic_scatter(dist_mat, delta_mat, out_dir / "scatter.png")
    stats.append(("scatter", ci, cd))

    # ② row‑mean
    ci, cd = plot_rowmean_scatter(dist_mat, delta_mat, out_dir / "scatter_binned.png")
    stats.append(("scatter_binned", ci, cd))

    # ③ sliding Δ
    ci, cd = plot_sliding_scatter(dist_mat, delta_mat, out_dir / "scatter_sliding.png",
                                   win=args.win, step=args.step)
    stats.append(("scatter_sliding", ci, cd))

    # ④ sliding accuracy
    perlogic_root = Path(f"results/output/perlogic/{lr_slug}")
    ci, cd = plot_accuracy_scatter(dist_mat, perlogic_root, out_dir / "scatter_acc.png",
                                   win=args.win, step=args.step)
    stats.append(("scatter_acc", ci, cd))

    # ── write per‑combo CSV (figure,inter,intra) ────────────────────────
    combo_df = pd.DataFrame(stats, columns=["figure", "inter", "intra"])
    combo_path = out_dir / "correlations.csv"
    combo_df.to_csv(combo_path, index=False)
    print(f"[info] saved per‑combo correlations → {combo_path}")

    # ── append to global aggregate ──────────────────────────────────────
    agg_df = combo_df.copy()
    agg_df.insert(0, "lr", args.lr)
    agg_df.insert(0, "seed", args.seed)
    agg_df.insert(0, "distance", args.distance)

    agg_path = Path("results/figures/correlations_all.csv")
    header_needed = not agg_path.exists()
    agg_df.to_csv(agg_path, mode="a", header=header_needed, index=False)
    print(f"[info] appended to global aggregate → {agg_path}\n")

    if args.plot_dist:
        # ⑤ distributions   (hist + KDE)
        plot_dist_vs_delta_hist(dist_mat, delta_mat,
                                out_dir / "hist_distance_delta.png")
        # ⑥ QQ-plot  distance vs delta
        plot_qq_dist_delta(dist_mat, delta_mat,
                        out_dir / "qq_distance_delta.png")



if __name__ == "__main__":
    main_cli()
