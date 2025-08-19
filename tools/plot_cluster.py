#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, os, re, math, shutil, uuid, itertools
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- helpers -----------------------------
LABEL0 = {0, 1, 3, 4, 8}  # aligns with plot_preliminary_intra_inter.py

def logic_labels(n: int) -> np.ndarray:
    return np.array([0 if (i % 10) in LABEL0 else 1 for i in range(n)], dtype=int)

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    G = X @ X.T
    sq = np.sum(X**2, axis=1, keepdims=True)
    D2 = sq + sq.T - 2*G
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, dtype=float)

def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b

# ---------------------- clustering metrics ------------------------
def silhouette_values(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    n = len(y)
    if n < 3 or len(np.unique(y)) < 2: return np.zeros(n, dtype=float)
    D = pairwise_distances(X)
    s = np.zeros(n, dtype=float)
    for i in range(n):
        same = (y == y[i]); other = ~same
        a = 0.0 if same.sum() <= 1 else D[i, same].sum() / (same.sum() - 1)
        b = 0.0 if other.sum() == 0 else D[i, other].mean()
        s[i] = 0.0 if (a == 0 and b == 0) else (b - a) / max(a, b)
    return s

def silhouette_mean(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    sv = silhouette_values(X, y)
    overall = float(np.nanmean(sv)) if sv.size else float("nan")
    m0 = float(np.nanmean(sv[y == 0])) if np.any(y == 0) else float("nan")
    m1 = float(np.nanmean(sv[y == 1])) if np.any(y == 1) else float("nan")
    return overall, m0, m1

def ch_score(X: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    n = len(y); k = len(np.unique(y))
    if k <= 1 or k >= n: return 0.0
    mu = X.mean(axis=0); S_W = 0.0; S_B = 0.0
    for lab in np.unique(y):
        idx = np.where(y == lab)[0]
        c = X[idx].mean(axis=0)
        S_W += np.sum((X[idx] - c)**2)
        S_B += len(idx) * np.sum((c - mu)**2)
    return safe_div(S_B / (k - 1), S_W / (n - k))

def db_score(X: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    labs = np.unique(y)
    if len(labs) <= 1: return 0.0
    cents, s = [], []
    for lab in labs:
        idx = np.where(y == lab)[0]
        c = X[idx].mean(axis=0); cents.append(c)
        d = np.sqrt(((X[idx] - c)**2).sum(axis=1)) if idx.size else np.array([])
        s.append(d.mean() if d.size else 0.0)
    C = pairwise_distances(np.stack(cents, axis=0))
    R = np.zeros_like(C)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if i == j: continue
            R[i, j] = safe_div(s[i] + s[j], C[i, j])
    return float(np.max(R, axis=1).mean())

# ------------------- point set reconstruction ---------------------
def masks_inter_intra(dist: pd.DataFrame, delta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match plot_basic_scatter():
      inter = off-diagonal & cross-label & (x!=0) & (y!=0)
      intra = diagonal only
    """
    n = len(dist)
    off_diag = ~np.eye(n, dtype=bool)
    diff_lbl = logic_labels(n)[:, None] != logic_labels(n)[None, :]
    x = dist.values.flatten(); y = delta.values.flatten()
    nonzero = (x != 0) & (y != 0)
    m_inter = (off_diag & diff_lbl).flatten() & nonzero
    m_intra = (~off_diag).flatten()
    return m_inter, m_intra

def points_scatter(dist: pd.DataFrame, delta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    m_inter, m_intra = masks_inter_intra(dist, delta)
    x = dist.values.flatten(); y = delta.values.flatten()
    Xi = np.c_[x[m_inter], y[m_inter]]; Xd = np.c_[x[m_intra], y[m_intra]]
    return Xi, Xd

def points_rowmean(dist: pd.DataFrame, delta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row-mean points:
      • Inter-pattern: for each row i, average over columns j where
        (j ≠ i) AND (label[j] ≠ label[i]) AND dist[i,j] != 0 AND delta[i,j] != 0 (both finite),
        producing one point (mean_x, mean_y) per row with any valid js.
      • Intra-pattern: diagonal pairs (dist[i,i], delta[i,i]).
    """
    n = len(dist)
    lbl = logic_labels(n)
    off_diag = ~np.eye(n, dtype=bool)
    diff_lbl = lbl[:, None] != lbl[None, :]

    D = dist.values
    A = delta.values
    nz = (D != 0) & (A != 0) & np.isfinite(D) & np.isfinite(A)

    vals_x, vals_y = [], []
    for i in range(n):
        js = np.where(off_diag[i] & diff_lbl[i] & nz[i])[0]
        if js.size:
            vals_x.append(D[i, js].mean())
            vals_y.append(A[i, js].mean())

    Xi = np.c_[np.array(vals_x, dtype=float), np.array(vals_y, dtype=float)] if vals_x else np.zeros((0, 2), dtype=float)
    Xd = np.c_[np.diag(D).astype(float), np.diag(A).astype(float)]
    return Xi, Xd

def points_sliding(dist: pd.DataFrame, delta: pd.DataFrame, win=0.10, step=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match plot_sliding_scatter():
      • Build masks as in plot_sliding_scatter (off-diagonal, cross-label, nonzero)
      • Slide fixed-width windows along x and average (x,y) within each bin
    """
    x, y = dist.values.flatten(), delta.values.flatten()
    n = len(dist)
    labels = logic_labels(n)

    off_diag = ~np.eye(n, dtype=bool)
    diff_lbl = labels[:, None] != labels[None, :]
    nonzero = (x != 0) & (y != 0)
    m_inter = (off_diag & diff_lbl).flatten() & nonzero
    m_intra = (~off_diag).flatten()

    def slide(xx: np.ndarray, yy: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros((0, 2))
        xx, yy = xx[mask], yy[mask]
        xs, ys = [], []
        lo, hi = float(xx.min()), float(xx.max())
        for left in np.arange(lo, hi - win + 1e-9, step):
            sel = (xx >= left) & (xx < left + win)
            if sel.any():
                xs.append(xx[sel].mean()); ys.append(yy[sel].mean())
        if not xs:
            return np.zeros((0, 2))
        return np.c_[np.array(xs), np.array(ys)]

    Xi = slide(x, y, m_inter)
    Xd = slide(x, y, m_intra)
    return Xi, Xd


def points_accuracy(dist: pd.DataFrame, perlogic_root: Path, win=0.10, step=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match plot_accuracy_scatter():
      • Read n_total.csv and delta_count.csv to form accuracy matrix
      • Masks: inter uses off-diagonal & cross-label & (x!=0) & ~nan(acc)
               intra uses diagonal & ~nan(acc)
      • Windowed (x, accuracy) with totals as weights inside each bin
    """
    path_tot, path_cor = perlogic_root / "n_total.csv", perlogic_root / "delta_count.csv"
    if not (path_tot.exists() and path_cor.exists()):
        return np.zeros((0, 2)), np.zeros((0, 2))

    mx = dist
    m_tot = pd.read_csv(path_tot, index_col=0).loc[mx.index, mx.columns]
    m_cor = pd.read_csv(path_cor, index_col=0).loc[mx.index, mx.columns]
    m_acc = m_cor.where(m_tot != 0) / m_tot

    x   = mx.values.flatten()
    acc = m_acc.values.flatten()
    tot = m_tot.values.flatten()

    n = len(mx)
    labels = logic_labels(n)
    off_diag = ~np.eye(n, dtype=bool)
    diff_lbl = labels[:, None] != labels[None, :]

    nonnan  = ~np.isnan(acc)
    m_inter = (off_diag & diff_lbl).flatten() & (x != 0) & nonnan
    m_intra = (~off_diag).flatten() & nonnan

    def window(xx: np.ndarray, a: np.ndarray, t: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return np.zeros((0, 2))
        xx, a, t = xx[mask], a[mask], t[mask]
        xs, ys = [], []
        lo, hi = float(xx.min()), float(xx.max())
        for left in np.arange(lo, hi - win + 1e-9, step):
            sel = (xx >= left) & (xx < left + win)
            if sel.any():
                numer, denom = float((a[sel] * t[sel]).sum()), float(t[sel].sum())
                xs.append(xx[sel].mean())
                ys.append(numer / denom if denom != 0 else 0.0)
        if not xs:
            return np.zeros((0, 2))
        return np.c_[np.array(xs), np.array(ys)]

    Xi = window(x, acc, tot, m_inter)
    Xd = window(x, acc, tot, m_intra)
    return Xi, Xd

# --------------------------- evaluation ----------------------------
def eval_points(label_intra: np.ndarray, label_inter: np.ndarray) -> Tuple[dict, np.ndarray, np.ndarray]:
    X = np.vstack([label_intra, label_inter]) if label_intra.size and label_inter.size else \
        (label_intra if label_inter.size == 0 else label_inter)
    if X.size == 0:
        return {"silhouette": float("nan"), "silhouette_intra": float("nan"),
                "silhouette_inter": float("nan"), "silhouette_x": float("nan"),
                "silhouette_intra_x": float("nan"), "silhouette_inter_x": float("nan"),
                "silhouette_y": float("nan"), "silhouette_intra_y": float("nan"),
                "silhouette_inter_y": float("nan"),
                "calinski_harabasz": float("nan"), "davies_bouldin": float("nan"),
                "n_intra": 0, "n_inter": 0}, X, np.array([])
    y = np.hstack([np.zeros(len(label_intra), dtype=int),
                   np.ones(len(label_inter), dtype=int)]) if label_intra.size and label_inter.size \
        else (np.zeros(len(label_intra), dtype=int) if label_inter.size == 0 else np.ones(len(label_inter), dtype=int))

    # 2D metrics
    s_all, s0, s1 = silhouette_mean(X, y)
    ch = ch_score(X, y); db = db_score(X, y)

    # 1D metrics per axis
    sx_all, sx0, sx1 = silhouette_mean(X[:, [0]], y)
    sy_all, sy0, sy1 = silhouette_mean(X[:, [1]], y)

    metrics = {"silhouette": s_all, "silhouette_intra": s0, "silhouette_inter": s1,
               "silhouette_x": sx_all, "silhouette_intra_x": sx0, "silhouette_inter_x": sx1,
               "silhouette_y": sy_all, "silhouette_intra_y": sy0, "silhouette_inter_y": sy1,
               "calinski_harabasz": ch, "davies_bouldin": db,
               "n_intra": int((y == 0).sum()), "n_inter": int((y == 1).sum())}
    return metrics, X, y

def grid_plot(Xy_by_figure: Dict[str, Tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    names = list(Xy_by_figure.keys()); k = len(names)
    cols = int(np.ceil(np.sqrt(k))); rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names):
        X, y = Xy_by_figure[name]
        if X.size == 0:
            ax.set_title(f"{name} (no data)"); ax.axis("off"); continue
        ax.scatter(X[:,0], X[:,1], c=y, s=10)
        ax.set_title(name); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def violin_by_distance(sil_samples: Dict[str, List[np.ndarray]], out_dir: Path, suffix: str = ""):
    """
    ONE violin per figure (combined silhouettes).
    - Adds a thick horizontal line + dot + text for mean; dashed line for median.
    Saves to: out_dir / f"violin_{fig_name}{suffix}.png"
    """
    for fig_name, chunks in sil_samples.items():
        if not chunks:
            continue
        values = np.concatenate(chunks)

        mean_val = float(np.nanmean(values))
        median_val = float(np.nanmedian(values))

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        parts = ax.violinplot([values], showmeans=False, showextrema=True, showmedians=True)
        ax.set_xticks([1]); ax.set_xticklabels(["all"])
        ax.set_ylabel("Silhouette (per-sample)")
        title_suffix = "combined" if suffix == "" else ("X only" if suffix == "_x" else "Y only")
        ax.set_title(f"Silhouette distribution — {fig_name} ({title_suffix})")

        # highlight mean (thick line + point + label)
        ax.hlines(mean_val, 0.85, 1.15, linewidth=4)
        ax.scatter([1], [mean_val], s=80, zorder=3)
        ax.annotate(f"mean={mean_val:.3f}", xy=(1, mean_val), xytext=(1.18, mean_val), va="center", ha="left")

        # optional median (thin dashed line + label)
        ax.hlines(median_val, 0.9, 1.1, linewidth=2, linestyles="--")
        ax.annotate(f"median={median_val:.3f}", xy=(1, median_val), xytext=(1.18, median_val), va="center", ha="left")

        fig.tight_layout()
        ax.figure.savefig(out_dir / f"violin_{fig_name}{suffix}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

# ----------------------- grouping / I-O utils ----------------------
def lr_to_slug(lr: float | str) -> str:
    s = str(lr)
    try:
        s = str(float(s))
    except Exception:
        pass
    return s.replace(".", "p").replace("-", "m")

def load_distance(distance: str, seed: int, tag: str) -> pd.DataFrame:
    f = Path(f"results/output/distance/{distance}/{tag}/seed{seed}.csv")
    return pd.read_csv(f, index_col=0)

def load_delta(lr: float | str, tag: str) -> pd.DataFrame:
    slug = lr_to_slug(lr)
    f = Path(f"results/output/perlogic/{slug}/{tag}/delta.csv")
    return pd.read_csv(f, index_col=0)

def load_acc_counts(lr: float | str, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    slug = lr_to_slug(lr)
    root = Path(f"results/output/perlogic/{slug}/{tag}")
    tot = pd.read_csv(root / "n_total.csv", index_col=0) if (root / "n_total.csv").exists() else pd.DataFrame()
    cor = pd.read_csv(root / "delta_count.csv", index_col=0) if (root / "delta_count.csv").exists() else pd.DataFrame()
    if tot.empty or cor.empty:
        return pd.DataFrame(), pd.DataFrame()
    return tot, cor

def block_diag_df(mats: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Block-diagonal concatenate square DataFrames."""
    if len(mats) == 1:
        return mats[0].copy()
    sizes = [m.shape[0] for m in mats]
    total = sum(sizes)
    big = np.zeros((total, total), dtype=float)
    idx_names: List[str] = []
    col_names: List[str] = []
    r0 = 0
    for k, m in enumerate(mats):
        n = m.shape[0]
        big[r0:r0+n, r0:r0+n] = m.values
        idx_names.extend([f"{ix}_r{k}" for ix in m.index])
        col_names.extend([f"{cx}_r{k}" for cx in m.columns])
        r0 += n
    return pd.DataFrame(big, index=idx_names, columns=col_names)

def parse_list(arg: str | None, cast) -> List:
    if arg is None:
        return []
    parts = [p.strip() for seg in arg.split(",") for p in seg.split()]
    out = []
    for p in parts:
        if p:
            try:
                out.append(cast(p))
            except Exception:
                out.append(p)
    return out

def group_runs(runs: List[Tuple[int, float]], k: Optional[int]) -> List[List[Tuple[int, float]]]:
    """
    Given list of (seed, lr), return groups:
      • k None/0 → one big group (default).
      • k == 1   → one group per run.
      • k >= 2   → all unordered k-combinations of runs.
    """
    if not k:
        return [runs]
    if k > len(runs):
        raise ValueError(f"--combine {k} but only {len(runs)} runs supplied")
    return list(itertools.combinations(runs, k))

# ------------------------------ main --------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate intra/inter clustering with grouped combinations (choose k from runs).")
    ap.add_argument("--distance", type=str, default="pot", help="Distance name (e.g., pot).")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9,10", help="Comma- or space-separated seeds, e.g. '0,1,2'.")
    ap.add_argument("--lrs", type=str, default="1e-5", help="Comma- or space-separated lrs, e.g. '1e-4 5e-5'.")
    ap.add_argument("--combine", type=int, default=10, help="Group size k: choose k from all (seed,lr) runs.")
    ap.add_argument("--subdir", type=str, default="origin", help="Tag (last layer) used in your outputs.")
    ap.add_argument("--out-root", type=str, default="results/figures/cluster", help="Root for outputs.")
    ap.add_argument("--win", type=float, default=0.10); ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    seeds = parse_list(args.seeds, int)
    lrs   = parse_list(args.lrs, float)
    if not seeds or not lrs:
        ap.error("Please provide --seeds and --lrs explicitly.")

    runs   = list(itertools.product(seeds, lrs))         # [(seed, lr), …]
    groups = group_runs(runs, args.combine)              # [[(seed,lr)…], …]

    # Aggregate CSV path
    agg_path = Path(args.out_root) / args.distance / "logs" / f"cluster_scores_{args.subdir}.csv"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not agg_path.exists()

    # For distance-level violin aggregation across all groups
    sil_samples_2d: Dict[str, List[np.ndarray]] = {"scatter": [], "rowmean": [], "sliding": [], "accuracy": []}
    sil_samples_x:  Dict[str, List[np.ndarray]] = {"scatter": [], "rowmean": [], "sliding": [], "accuracy": []}
    sil_samples_y:  Dict[str, List[np.ndarray]] = {"scatter": [], "rowmean": [], "sliding": [], "accuracy": []}

    for gidx, grp in enumerate(groups, 1):
        seeds_grp = sorted({s for s, _ in grp})
        lrs_grp   = sorted({lr for _, lr in grp})
        lr_slugs  = [lr_to_slug(lr) for lr in lrs_grp]

        # Build block-diag matrices for this group (distance paired with seed, delta paired with lr)
        dist_cat  = block_diag_df([load_distance(args.distance, s, args.subdir) for s, _ in grp])
        delta_cat = block_diag_df([load_delta(lr, args.subdir)                  for _, lr in grp])

        # Accuracy counts block-diag (if present)
        tot_blocks, cor_blocks = [], []
        for _, lr in grp:
            tot, cor = load_acc_counts(lr, args.subdir)
            if not tot.empty and not cor.empty:
                tot_blocks.append(tot)
                cor_blocks.append(cor)
        tot_cat = block_diag_df(tot_blocks) if tot_blocks else pd.DataFrame()
        cor_cat = block_diag_df(cor_blocks) if cor_blocks else pd.DataFrame()

        # If accuracy counts exist, write to a temp root for window function
        tmp_root = None
        if not tot_cat.empty and not cor_cat.empty:
            tmp_root = Path(f"results/output/perlogic/__tmp_{uuid.uuid4().hex}")
            tmp_root.mkdir(parents=True, exist_ok=True)
            (tmp_root / "n_total.csv").write_text(tot_cat.to_csv())
            (tmp_root / "delta_count.csv").write_text(cor_cat.to_csv())

        # point clouds from combined matrices
        Xi_sc, Xd_sc = points_scatter(dist_cat, delta_cat)
        Xi_rm, Xd_rm = points_rowmean(dist_cat, delta_cat)
        Xi_sl, Xd_sl = points_sliding(dist_cat, delta_cat, win=args.win, step=args.step)
        Xi_ac, Xd_ac = points_accuracy(dist_cat, tmp_root if tmp_root else Path("."), win=args.win, step=args.step)

        metrics_rows = []; per_fig_points = {}
        for fig_name, (Xi, Xd) in {
            "scatter": (Xi_sc, Xd_sc),
            "rowmean": (Xi_rm, Xd_rm),
            "sliding": (Xi_sl, Xd_sl),
            "accuracy": (Xi_ac, Xd_ac),
        }.items():
            m, X, y = eval_points(Xd, Xi)  # class 0=intra, 1=inter

            # per-dimension silhouettes
            if X.size:
                sx_all, sx0, sx1 = silhouette_mean(X[:, [0]], y)
                sy_all, sy0, sy1 = silhouette_mean(X[:, [1]], y)
                m["silhouette_x"], m["silhouette_intra_x"], m["silhouette_inter_x"] = sx_all, sx0, sx1
                m["silhouette_y"], m["silhouette_intra_y"], m["silhouette_inter_y"] = sy_all, sy0, sy1

                sil_samples_2d[fig_name].append(silhouette_values(X, y))
                sil_samples_x[fig_name].append(silhouette_values(X[:, [0]], y))
                sil_samples_y[fig_name].append(silhouette_values(X[:, [1]], y))

            m.update({"figure": fig_name,
                      "distance": args.distance,
                      "seeds": ";".join(map(str, seeds_grp)),
                      "lrs": ";".join(map(str, lrs_grp)),
                      "group_size": len(grp)})
            metrics_rows.append(m)

            per_fig_points[fig_name] = (X, y)

        # Output dir: results/figures/<distance>/mSEEDS/mLRS/<subdir>/group_{gidx}
        seed_tag = "m" + "-".join(map(str, seeds_grp))
        lr_tag   = "m" + "-".join(lr_slugs)
        out_dir  = Path(args.out_root) / args.distance / seed_tag / lr_tag / args.subdir / f"group_{gidx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write per-group CSV and grid
        pd.DataFrame(metrics_rows).to_csv(out_dir / "cluster_scores.csv", index=False)
        grid_plot(per_fig_points, out_dir / "scatter_grid.png")

        # Append to aggregate (include per-dim cols when可用)
        cols = ["distance","seeds","lrs","group_size","figure",
                "silhouette","silhouette_intra","silhouette_inter",
                "silhouette_x","silhouette_intra_x","silhouette_inter_x",
                "silhouette_y","silhouette_intra_y","silhouette_inter_y",
                "calinski_harabasz","davies_bouldin","n_intra","n_inter"]
        df = pd.DataFrame(metrics_rows)
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(agg_path, mode="a", header=header_needed, index=False); header_needed = False

        # Cleanup temp
        if tmp_root:
            shutil.rmtree(tmp_root, ignore_errors=True)

        print(f"[✓] group {gidx}: seeds={seeds_grp} lrs={lrs_grp} → {out_dir}")

    # Distance-level violins: 2D (combined), X-only, Y-only
    vdir = Path(args.out_root) / args.distance / "violin" / args.subdir
    vdir.mkdir(parents=True, exist_ok=True)
    violin_by_distance(sil_samples_2d, vdir, suffix="")
    violin_by_distance(sil_samples_x,  vdir, suffix="_x")
    violin_by_distance(sil_samples_y,  vdir, suffix="_y")
    print(f"[✓] violin plots written under {vdir} (2D, X-only, Y-only)")

if __name__ == "__main__":
    main()