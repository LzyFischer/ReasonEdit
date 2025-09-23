#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grouped bar plot (no external files): three datasets (Level 1/2/3) × methods.
Legend order: raw, LoRA, BIMT, AlphaEdit, ROME, Ours.

Usage
-----
python plot_cluster_bars_fixed.py --out /path/to/cluster_bar_levels.png
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

# ─────────── Global style ───────────
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    "axes.facecolor": "#EEF0F2",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.color": "gray",
    "grid.linewidth": 1.5,
    "grid.alpha": 0.5,
    "font.size": 22,
})

# 7-color palette（与原脚本风格接近）
PALETTE = [
    (123/255.0, 141/255.0, 191/255.0),
    (250/255.0, 127/255.0, 111/255.0),
    (130/255.0, 176/255.0, 210/255.0),
    (255/255.0, 190/255.0, 122/255.0),
    (87/255.0, 184/255.0, 147/255.0),
    "#C76DA2",
]

# ─────────── Input (fill here) ───────────
# 方法顺序（图例顺序）
METHODS = ["Raw", "REdit (w/o Edit)"]

# 你的三组数据（我将每组第二行为“标准差”来用）
# 若这些是“方差”，把 use_std=False（代码里会开方）。
DATA = {
    "Level 1": {
        "mean": [60.7, 66.3],             # ROME 缺失 -> None
        "std": [2.3, 2.1],   # 同上
    },
    "Level 2": {
        "mean": [53.2, 57.1],
        "std": [1.4, 0.8],
    },
    "Level 3": {
        "mean": [45.1, 47.9],
        "std": [1.6, 0.8],
    },
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="输出图片路径（默认：当前目录 cluster_bar_levels.png）")
    ap.add_argument("--ylabel", default="Score (%)", help="Y 轴标签")
    ap.add_argument("--title", default="", help="标题（可留空）")
    ap.add_argument("--use_std", action="store_true",
                    help="将第二行数值视为标准差（默认：视为标准差；若给的是方差请不要加该参数，而改成 --as_var）。")
    ap.add_argument("--as_var", action="store_true",
                    help="将第二行数值视为方差（会自动开方得到 std）。")
    args = ap.parse_args()

    # 兼容：默认按“标准差”理解；若指定 --as_var 则开方
    use_std = True
    if args.as_var:
        use_std = False

    datasets = list(DATA.keys())  # ["Level 1", "Level 2", "Level 3"]
    n_sets = len(datasets)
    x = np.arange(n_sets)

    # 统计每个方法是否有完整数据
    method_has_data = []
    for mi, m in enumerate(METHODS):
        ok = True
        for ds in datasets:
            mean_row = DATA[ds]["mean"]
            std_row  = DATA[ds]["std"]
            if mi >= len(mean_row):
                ok = False
                break
            if mean_row[mi] is None or std_row[mi] is None:
                ok = False
                break
        method_has_data.append(ok)

    if not any(method_has_data):
        raise RuntimeError("所有方法数据都缺失，请检查 DATA 填写。")

    # 为可视方法分配颜色 & 序号
    visible_methods = [m for m, ok in zip(METHODS, method_has_data) if ok]
    if len(visible_methods) < len(METHODS):
        missing = [m for m, ok in zip(METHODS, method_has_data) if not ok]
        print(f"[WARN] 以下方法数据缺失，已跳过：{', '.join(missing)}")

    n_types = len(visible_methods)
    bar_w = min(0.8 / max(n_types, 1), 0.28)

    fig, ax = plt.subplots(figsize=(10, 10 * 0.58))

    # 逐方法画柱
    for i, m in enumerate(visible_methods):
        offset = (i - (n_types - 1) / 2) * bar_w
        means = []
        stds  = []
        mi = METHODS.index(m)
        for ds in datasets:
            v_mean = DATA[ds]["mean"][mi]
            v_std = DATA[ds]["std"][mi]
            if v_mean is None or v_std is None:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(v_mean)
                stds.append(v_std if use_std else (v_std ** 0.5))

        # 区分 Ours（加 hatch）
        hatch = '//' if 're' in m.lower() else None
        ax.bar(
            x + offset, means, bar_w, yerr=stds, capsize=5,
            label=m, color=PALETTE[i % len(PALETTE)],
            linewidth=2.5, edgecolor="black", alpha=0.85,
            error_kw=dict(linewidth=2.0),
            hatch=hatch
        )

        # 可选：在柱顶标注数值（保留 1 位小数）
        # for xi, (yy, ee) in enumerate(zip(means, stds)):
        #     if np.isfinite(yy):
        #         ax.text(x[xi] + offset, yy + (ee if np.isfinite(ee) else 0) + 0.4,
        #                 f"{yy:.1f}", ha="center", va="bottom", fontsize=14, rotation=0)

    # 轴与图例
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=22)
    ax.set_ylabel(args.ylabel, fontsize=24)
    if args.title:
        ax.set_title(args.title, fontsize=22)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.4),
        ncol=3,
        frameon=True
    )

    # Y 轴下限略高一点，便于区分
    ax.set_ylim(bottom=max(0, min(50, ax.get_ylim()[0])))
    # 美化边框
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_linewidth(1.5)
        ax.spines[side].set_edgecolor("black")

    fig.tight_layout()

    out = Path(args.out) if args.out else Path("./cluster_bar_levels.png")
    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {out_pdf}")

if __name__ == "__main__":
    main()