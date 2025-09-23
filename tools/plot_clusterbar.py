#!/usr/bin/env python3
"""
Grouped bar plot for cluster metrics with one chosen figure type and one metric family.

Examples
--------
python plot_cluster_bars.py \
  --dir /scratch/vjd5zr/project/ReasonEdit/results/figures/cluster/pot/logs \
  --figure rowmean \
  --metric silhouette \
  --labels origin=Origin,reptile_00045=Reptile(45),reptile_d_00040=Reptile-D(40)

Notes
-----
• Supports both "wide" CSVs (columns like silhouette, silhouette_x, silhouette_y)
  and "long" CSVs with columns like ['figure','metric','value', ...].
• Draws mean±std across rows after filtering by --figure.
• X-ticks are always [<metric>, <metric>_x, <metric>_y].
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, List, Tuple
from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────── GLOBAL STYLE ────────────────────
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "axes.facecolor": "#EEF0F2",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "gray",
    "grid.linewidth": 1.5,
    "grid.alpha": 0.5,
})
plt.rcParams["font.size"] = 22

# Nice 7-color palette (pastel + vibrant mix)
PALETTE = [
    (123/255.0, 141/255.0, 191/255.0),  # warm coral
    (250/255.0, 127/255.0, 111/255.0),  # mustard/golden
    (130/255.0, 176/255.0, 210/255.0),  # green
    (255/255.0, 190/255.0, 122/255.0),  # teal green
    (87/255.0, 184/255.0, 147/255.0),  # bright blue
    "#C76DA2",  # pink/magenta
]


def parse_labels(arg: str | None) -> Dict[str, str]:
    """Parse --labels like 'origin=Origin,reptile_00045=Reptile(45)'."""
    if not arg:
        return {}
    out = {}
    for pair in arg.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def file_type_name(path: Path) -> str:
    """Derive a type key from file name: cluster_scores_<TYPE>.csv."""
    name = path.stem  # cluster_scores_origin
    return re.sub(r"^cluster_scores_", "", name)


def to_wide_three_metrics(df: pd.DataFrame, metric_family: str) -> pd.DataFrame:
    """
    Return a DataFrame with exactly three columns:
    [metric, metric_x, metric_y] for the chosen family.
    Works for both wide and long formats.
    """
    m0, m1, m2 = metric_family, f"{metric_family}_x", f"{metric_family}_y"
    cols = {c.lower(): c for c in df.columns}

    # Case A: already wide (columns exist)
    if m0 in cols and m1 in cols and m2 in cols:
        return df[[cols[m0], cols[m1], cols[m2]]].astype(float).rename(
            columns={cols[m0]: m0, cols[m1]: m1, cols[m2]: m2}
        )

    # Case B: long form with 'metric' and a value column
    metric_col = None
    value_col = None
    for cand in ["metric", "measure", "name"]:
        if cand in cols:
            metric_col = cols[cand]
            break
    # Value column heuristics
    for cand in ["value", "score", "val", "mean", "avg"]:
        if cand in cols:
            value_col = cols[cand]
            break
    if value_col is None:
        # Fallback: last numeric column
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            value_col = num_cols[-1]

    if metric_col is not None and value_col is not None:
        keep = df[df[metric_col].str.lower().isin([m0, m1, m2])]
        wide = keep.pivot_table(index=keep.index, columns=metric_col, values=value_col, aggfunc="mean")
        # Guarantee all three exist
        for m in [m0, m1, m2]:
            if m not in wide.columns:
                wide[m] = np.nan
        return wide[[m0, m1, m2]]

    raise KeyError(f"Could not find {m0},{m1},{m2} as columns or via long-form ('metric','value')")


def filter_figure(df: pd.DataFrame, figure_type: str) -> pd.DataFrame:
    """Filter by the 'figure' column if present; else return unchanged."""
    match_col = None
    for cand in ["figure", "fig", "mode", "type"]:
        if cand in df.columns.str.lower().tolist():
            match_col = df.columns[df.columns.str.lower() == cand][0]
            break
    if match_col is None:
        return df
    mask = df[match_col].astype(str).str.lower() == figure_type.lower()
    # If mask empties the frame, fallback to unchanged to avoid surprises
    return df[mask] if mask.any() else df


def collect_stats_per_file(
    csv: Path, metric_family: str, figure_type: str
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Return (type_key, means[3], stds[3]) for the three metrics."""
    df = pd.read_csv(csv)
    df = filter_figure(df, figure_type)
    wide = to_wide_three_metrics(df, metric_family)
    means = wide.mean(axis=0, skipna=True).to_numpy()
    stds = wide.std(axis=0, ddof=1, skipna=True).to_numpy()
    return file_type_name(csv), means, stds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/results/figures/cluster/level_1/edit/logs", help="Folder containing cluster_scores_*.csv")
    ap.add_argument("--figure", default='rowmean', help="One figure type to include (e.g., rowmean, scatter, sliding, accuracy)")
    ap.add_argument("--metric", default="silhouette", choices=["silhouette", "acc", "nmi", "ari"], help="Metric family to plot")
    ap.add_argument("--labels", default="nometa_00045=w/o MCL, reptile_d_00040=w/o NSP, reptile_nsm_00040=w/o PDP, reptile_nsd_00045=REdit, origin=Raw", help="Legend mapping: key=Label,key2=Label2")
    ap.add_argument("--order", default="origin, nometa_00045, reptile_d_00040, reptile_nsm_00040, reptile_nsd_00045", help="Comma list of type keys to control order (e.g., origin,reptile_00045,...)")
    ap.add_argument("--out", default="", help="Output PNG path (default: <dir>/cluster_bar_<metric>_<figure>.png)")
    args = ap.parse_args()

    base = Path(args.dir)
    csvs = sorted(base.glob("cluster_scores_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs found in {base}")

    label_map = parse_labels(args.labels)
    wanted_order = [s.strip() for s in args.order.split(",") if s.strip()]

    # Collect stats
    records = []
    for csv in csvs:
        try:
            key, means, stds = collect_stats_per_file(csv, args.metric, args.figure)
            records.append((key, means, stds))
        except Exception as e:
            print(f"[WARN] {csv.name}: {e}")

    if not records:
        raise RuntimeError("No usable CSVs after parsing.")

    # Order: explicit --order > put 'origin' first if present > filename order
    keys = [r[0] for r in records]
    if wanted_order:
        key_order = wanted_order + [k for k in keys if k not in wanted_order]
    elif "origin" in keys:
        key_order = ["origin"] + [k for k in keys if k != "origin"]
    else:
        key_order = keys

    # Prepare arrays in final order
    metrics = [args.metric, f"{args.metric}_x", f"{args.metric}_y"]
    ordered = [(k, *next((m, s) for kk, m, s in records if kk == k)) for k in key_order]

    # ───────────── Plot ─────────────
    x = np.arange(3)  # three ticks
    n_types = len(ordered)
    bar_w = min(0.8 / max(n_types, 1), 0.28)

    fig, ax = plt.subplots(figsize=(10, 10*0.58))
    for i, (k, means, stds) in enumerate(ordered):
        offset = (i - (n_types - 1) / 2) * bar_w
        label = label_map.get(k, k)  # rename if provided
        if i == 4:
            # with hatch
            ax.bar(x + offset, means, bar_w, yerr=stds, capsize=5, label=label, color=PALETTE[i % len(PALETTE)], linewidth=2.5, edgecolor="black", alpha=0.8, hatch='//', error_kw=dict(linewidth=2.5))
        else:
            ax.bar(x + offset, means, bar_w, yerr=stds, capsize=5, label=label, color=PALETTE[i % len(PALETTE)], linewidth=2.5, edgecolor="black", alpha=0.8, error_kw=dict(linewidth=2.5))

    ax.set_xticks(x)
    ax.set_xticklabels(['Overall', 'Distance', 'Interference'], fontsize=26)
    ylabel = f"Silouette Score"
    ax.set_ylabel(ylabel, fontsize=26)
    # ax.set_title(f"{args.metric.capitalize()} – {args.figure}  |  " + ", ".join([label_map.get(k, k) for k in key_order]))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.83, 1),  # move it above the plot
        ncol=1,          # put all entries in a single row
        # frameon=False
    )
    # these metric families are 0–1; clamp for clean comparison
    # ylim bottom at 0 but auto top
    ax.set_ylim(bottom=0.10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)   # thickness
        spine.set_edgecolor("black")  # color
        # remove top and right spines
        if spine.spine_type in ['top', 'right']:
            spine.set_visible(False)

    fig.tight_layout()
    out = Path(args.out) if args.out else base / f"cluster_bar_{args.metric}_{args.figure}.png"
    out_pdf = out.with_suffix('.pdf')
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=600)
    print("Saved plot to:", out)


if __name__ == "__main__":
    main()