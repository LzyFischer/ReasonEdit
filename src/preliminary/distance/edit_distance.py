from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR
#!/usr/bin/env python3
"""
plot_weighted_edit_distance.py
────────────────────────────────────────────────────────────────────────────
1. 读取 circuit-mask JSON (logic_###_splitS_part[A|B].json)
2. 把每个掩码展平成 {edge: normalised_weight} 字典
3. 依照 3-split averaging 规则计算加权 L₁ 距离
4. (可选) 根据抽象公式距离做层次聚类重新排序
5. 绘制热图

Author : <you>
Date   : 2025-06-14
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# ─────────────────────────────────────────────── regex & util constants ──
MASK_RE = re.compile(r"(logic_\d{3})_split(\d+)_part([AB])\.json$")

# ═════════════════════════════ 1. 低阶工具 ════════════════════════════════
def load_weight_dict(path: Path, mode: str = "sum") -> Dict[str, float]:
    """
    JSON → {edge: normalised_weight}
    mode:
      "sum" :  w / Σw
      "max" :  w / max(w)
    """
    raw = json.loads(path.read_text())
    flat: Dict[str, float] = {}

    def rec(name: str, arr):
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
        s = sum(flat.values()); flat = {k: v / s for k, v in flat.items()}
    elif mode == "max":
        m = max(flat.values()); flat = {k: v / m for k, v in flat.items()}
    return flat


def w_edit(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    """L1 distance on union of keys."""
    keys = set(d1) | set(d2)
    return sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)


# ═════════════════════════════ 2. 读入 & 组织数据 ═════════════════════════
def collect_weights(root_pattern: str) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """data[logic][split][part] = weight_dict."""
    if any(ch in root_pattern for ch in "*?[]"):
        files = sorted(Path().glob(root_pattern))
    else:
        root  = Path(root_pattern)
        files = sorted(root.rglob("logic_*_split43_part*.json"))

    if not files:
        raise RuntimeError(f"No JSON files found under: {root_pattern}")

    data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for f in files:
        if (m := MASK_RE.match(f.name)):
            logic, split, part = m.groups()
            data[logic][split][part] = load_weight_dict(f, mode="sum")
    return data


# ═════════════════════════════ 3. 加权编辑距离矩阵 ════════════════════════
def build_wedit_matrix(data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]
                       ) -> tuple[np.ndarray, List[str]]:
    logics = sorted(data.keys())
    n      = len(logics)
    mat    = np.zeros((n, n))

    for i, li in enumerate(logics):
        for j, lj in enumerate(logics):
            if i == j:
                dists = [
                    w_edit(parts["A"], parts["B"])
                    for parts in data[li].values() if {"A", "B"} <= parts.keys()
                ]
                mat[i, j] = np.mean(dists) if dists else 0.0
            else:
                split_vals = []
                for s in set(data[li]) & set(data[lj]):      # 共同 split
                    Ai = data[li][s].get("A"); Bi = data[li][s].get("B")
                    Aj = data[lj][s].get("A"); Bj = data[lj][s].get("B")
                    combos = []
                    if Ai and Bj: combos.append(w_edit(Ai, Bj))
                    if Bi and Aj: combos.append(w_edit(Bi, Aj))
                    if Ai and Aj: combos.append(w_edit(Ai, Aj))
                    if Bi and Bj: combos.append(w_edit(Bi, Bj))
                    if combos:
                        split_vals.append(np.mean(combos))
                mat[i, j] = np.mean(split_vals) if split_vals else 0.0
    return mat, logics


# ═══════════════════════════ 4. (可选) 抽象聚类排序 ═══════════════════════
# 与前两个脚本相同：硬编码示例，可改为读取文件
ASSIGN_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s+is\s+(True|False)", re.I)
RULE_RE   = re.compile(r"\(\s*(.+?)\s*\)\s*->")

def load_abstract_formulas() -> Dict[str, str]:
    return {
        "logic_000": "aaa is True.\naab is False.\n(aaa or aab) -> aac.",
        "logic_001": "aab is True.\naaa is True.\n(aaa or aab) -> aac.",
        "logic_002": "aaa is True.\n(NOT aaa) -> aab.",
        "logic_003": "aab is True.\naaa is True.\n(aaa and aab) -> aac.",
    }

def parse_formula(text: str):
    var_vals, ant_vars, op = {}, set(), None
    for line in text.splitlines():
        if m := ASSIGN_RE.match(line):
            var_vals[m.group(1)] = (m.group(2).lower() == "true")
        elif m := RULE_RE.search(line):
            antecedent = m.group(1).lower()
            if " or " in antecedent:
                op = "or";  ant_vars.update(x.strip() for x in antecedent.split(" or "))
            elif " and " in antecedent:
                op = "and"; ant_vars.update(x.strip() for x in antecedent.split(" and "))
            elif "not " in antecedent:
                op = "not"; ant_vars.add(antecedent.split()[-1].strip())
            else:
                op = "atom"; ant_vars.add(antecedent.strip())
    return var_vals, ant_vars, op

def abstract_distance(f1: str, f2: str) -> int:
    v1, ant1, op1 = parse_formula(f1)
    v2, ant2, op2 = parse_formula(f2)
    all_vars     = set(v1) | set(v2)
    truth_diff   = sum(v1.get(k) != v2.get(k) for k in all_vars if k in v1 and k in v2)
    missing_diff = sum(1 for k in all_vars if (k in v1) ^ (k in v2))
    ant_num_diff = abs(len(ant1) - len(ant2))
    op_diff      = 0 if op1 == op2 else 1
    return truth_diff + missing_diff + ant_num_diff + op_diff

def reorder_by_abstract(mat: np.ndarray, labels: List[str]) -> tuple[np.ndarray, List[str]]:
    abs_formulas = load_abstract_formulas()
    n = len(labels)
    abs_mat = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            abs_mat[i, j] = abstract_distance(abs_formulas.get(li, ""),
                                              abs_formulas.get(lj, ""))
    order = sch.leaves_list(sch.linkage(abs_mat, method="average"))[::-1]
    return mat[order][:, order], [labels[i] for i in order]


# ═════════════════════════════ 5. 绘图 ═══════════════════════════════════
def plot_heatmap(mat: np.ndarray, labels: List[str], out_png: Path,
                 cmap: str = "cividis", block: int = 3):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n*0.5), max(5, n*0.5)))
    im = ax.imshow(mat, cmap=cmap)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise Weighted-Edit Distance (3-split avg)")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"✓ Saved heat-map → {out_png}")


# ═════════════════════════════ 6. CLI ════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(OUTPUTS_DIR / "attr_scores/qwen2_5_3b_instruct/5"),
                   help="Directory or glob of *.json mask files")
    p.add_argument("--block", type=int, default=3,
                   help="Dashed grid every N cells")
    p.add_argument("--no_cluster", action="store_true",
                   help="Skip abstract-distance reordering")
    return p

def main():
    args = build_parser().parse_args()
    out_png = Path(args.input) / "weighted_edit_distance_matrix.png"

    data           = collect_weights(args.input)
    wedit_mat, lbl = build_wedit_matrix(data)

    pd.DataFrame(wedit_mat, index=lbl, columns=lbl) \
      .to_csv(Path(args.input) / "weighted_edit_distance_raw.csv", float_format="%.6f")
    print("✓ Saved raw matrix →", Path(args.input) / "weighted_edit_distance_raw.csv")
    # ─────────────────────────────────────────────────────────────────────

    if not args.no_cluster:
        wedit_mat, lbl = reorder_by_abstract(wedit_mat, lbl)

    plot_heatmap(wedit_mat, lbl, out_png, block=args.block)


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()