#!/usr/bin/env python3
"""
plot_logic_distance.py
————————————————————————————————————————————————————————
1. 读取 mask JSON (logic_###_splitS_part[A|B].json)
2. 计算逻辑间 Jaccard 距离 (根据 3-split 规则)
3. 解析抽象公式 -> 生成自定义距离并用层次聚类重新排序行列
4. 绘制热图

Author : <you>
Date   : 2025-06-14
"""

from __future__ import annotations
import argparse
import json
import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List
import pdb
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

# ───────────────────────────────────────────────────────────── constants ──
MASK_NAME_RE = re.compile(r"(logic_\d{3})_split(\d+)_part([AB])\.json$")
ASSIGN_RE    = re.compile(r"^\s*([a-zA-Z_]\w*)\s+is\s+(True|False)", re.I)
RULE_RE      = re.compile(r"\(\s*(.+?)\s*\)\s*->")


# ═════════════════════════════════════════ 1. 低阶工具 ════════════════════
def load_positive_edges(path: Path, thr: float = 1e-5) -> Set[str]:
    """Flatten nested list‐like JSON into a set of edge names whose score > thr."""
    raw = json.loads(path.read_text())
    edges: Set[str] = set()

    def rec(name: str, arr):
        if isinstance(arr, (int, float)):
            if arr > thr:
                edges.add(name)
        else:
            for i, v in enumerate(arr):
                rec(f"{name}:{i}", v)

    for mod, arr in raw.items():
        rec(mod, arr)
    return edges


def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    """1 − IoU."""
    u = a | b
    return 0.0 if not u else 1.0 - len(a & b) / len(u)


# ══════════════════════════════════════ 2. 组织数据结构 ═══════════════════
def collect_masks(root_pattern: str) -> Dict[str, Dict[str, Dict[str, Set[str]]]]:
    """
    返回 data[logic][split][part] = edge_set
    `root_pattern` 可以是包含 *?[] 的 glob，也可以是目录。
    """
    if any(ch in root_pattern for ch in "*?[]"):
        files = sorted(Path().glob(root_pattern))
    else:
        root = Path(root_pattern)
        files = sorted(root.rglob("logic_*_split44_part*.json"))

    if not files:
        raise RuntimeError("No JSON files found under:", root_pattern)

    data: Dict[str, Dict[str, Dict[str, Set[str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for f in files:
        m = MASK_NAME_RE.match(f.name)
        if not m:
            continue
        logic, split, part = m.groups()
        data[logic][split][part] = load_positive_edges(f)
    return data


# ═══════════════════════════════════ 3. 计算 Jaccard 距离矩阵 ═════════════
def build_jaccard_matrix(data: Dict[str, Dict[str, Dict[str, Set[str]]]]
                          ) -> tuple[np.ndarray, List[str]]:
    logics = sorted(data.keys())
    n      = len(logics)
    mat    = np.zeros((n, n))

    for i, li in enumerate(logics):
        for j, lj in enumerate(logics):
            if i == j:                                   # 同一 logic
                dists = [
                    jaccard_distance(parts["A"], parts["B"])
                    for parts in data[li].values()
                    if "A" in parts and "B" in parts
                ]
                mat[i, j] = np.mean(dists) if dists else 0.0
            else:                                        # 不同 logic
                split_dists = []
                for s in set(data[li]) & set(data[lj]):  # 共同 split
                    Ai = data[li][s].get("A"); Bi = data[li][s].get("B")
                    Aj = data[lj][s].get("A"); Bj = data[lj][s].get("B")
                    combo = []
                    if Ai and Bj: combo.append(jaccard_distance(Ai, Bj))
                    if Bi and Aj: combo.append(jaccard_distance(Bi, Aj))
                    if Ai and Aj: combo.append(jaccard_distance(Ai, Aj))
                    if Bi and Bj: combo.append(jaccard_distance(Bi, Bj))
                    if combo:
                        split_dists.append(np.mean(combo))
                mat[i, j] = np.mean(split_dists) if split_dists else 0.0
    return mat, logics


# ═══════════════════════════════ 4. 抽象公式解析与距离 ════════════════════
def parse_formula(text: str):
    """→ (var_value_map, antecedent_vars:set, operator:str)"""
    var_vals, ant_vars, op = {}, set(), None
    for line in text.splitlines():
        if m := ASSIGN_RE.match(line):
            var_vals[m.group(1)] = (m.group(2).lower() == "true")
            continue
        if m := RULE_RE.search(line):
            antecedent = m.group(1).lower()
            if " or " in antecedent:
                op = "or"
                ant_vars.update(x.strip() for x in antecedent.split(" or "))
            elif " and " in antecedent:
                op = "and"
                ant_vars.update(x.strip() for x in antecedent.split(" and "))
            elif "not " in antecedent:
                op = "not"
                ant_vars.add(antecedent.split()[-1].strip())
            else:
                op = "atom"
                ant_vars.add(antecedent.strip())
    return var_vals, ant_vars, op


def abstract_distance(f1: str, f2: str) -> int:
    v1, ant1, op1 = parse_formula(f1)
    v2, ant2, op2 = parse_formula(f2)

    all_vars      = set(v1) | set(v2)
    truth_diff    = sum(v1.get(x) != v2.get(x) for x in all_vars if x in v1 and x in v2)
    missing_diff  = sum(1 for x in all_vars if (x in v1) ^ (x in v2))
    ant_num_diff  = abs(len(ant1) - len(ant2))
    op_diff       = 0 if op1 == op2 else 1
    return truth_diff + missing_diff + ant_num_diff + op_diff


def load_abstract_formulas() -> Dict[str, str]:
    """
    TODO: 改成从文件系统读取。
    目前仅供演示的 4 条硬编码示例。
    """
    return {
        "logic_000": "aaa is True.\naab is False.\n(aaa or aab) -> aac.\nDeduce the result of aac.",
        "logic_001": "aab is True.\naaa is True.\n(aaa or aab) -> aac.\nDeduce the result of aac.",
        "logic_002": "aaa is True.\n(NOT aaa) -> aab.\nDeduce the result of aab.",
        "logic_003": "aab is True.\naaa is True.\n(aaa and aab) -> aac.\nDeduce the result of aac.",
    }


def reorder_by_abstract(mat: np.ndarray, logics: List[str]) -> tuple[np.ndarray, List[str]]:
    """层次聚类获得排序索引，然后可再自行 reverse。"""
    abs_formulas = load_abstract_formulas()
    n = len(logics)
    abs_mat = np.zeros((n, n))
    for i, li in enumerate(logics):
        for j, lj in enumerate(logics):
            abs_mat[i, j] = abstract_distance(
                abs_formulas.get(li, ""), abs_formulas.get(lj, "")
            )
    
    linkage = sch.linkage(abs_mat, method="average")
    order   = sch.leaves_list(linkage)[::-1]      # 反转叶序
    return mat[order][:, order], [logics[i] for i in order]


# ═══════════════════════════════════════ 5. 绘图 ═════════════════════════
def plot_heatmap(mat: np.ndarray, labels: List[str], out_png: Path, block: int = 3):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * .5), max(5, n * .5)))
    im = ax.imshow(mat, cmap="viridis")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise Jaccard Distance (3-split avg)")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"✓ Saved heat-map → {out_png}")


# ═══════════════════════════════════════ 6. CLI ══════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input",
                   default="output/attr_scores/gpt2/0_05",
                   help="Directory or glob for *.json mask files")
    p.add_argument("--block", type=int, default=3,
                   help="Add dashed grid line every N cells")
    p.add_argument("--no_cluster", action="store_true",
                   help="Skip abstract-distance reordering")
    return p


def main():
    args = build_parser().parse_args()
    out_png = Path(args.input) / "jaccard_matrix.png"

    data                = collect_masks(args.input)
    jaccard_mat, labels = build_jaccard_matrix(data)

    raw_csv = Path(args.input) / "jaccard_distance_raw.csv"
    pd.DataFrame(jaccard_mat, index=labels, columns=labels) \
      .to_csv(raw_csv, float_format="%.6f")
    print("✓ Saved raw matrix →", raw_csv)
    # ─────────────────────────────────────────────────────────────────────


    if not args.no_cluster:
        jaccard_mat, labels = reorder_by_abstract(jaccard_mat, labels)

    plot_heatmap(jaccard_mat, labels, out_png, block=args.block)


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()