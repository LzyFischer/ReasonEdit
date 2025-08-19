from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR
#!/usr/bin/env python3
"""
plot_ot_distance.py
────────────────────────────────────────────────────────────────────────────
1. 读取 circuit-mask JSON (logic_###_splitS_part[A|B].json)
2. 将每个掩码展平成 bag-of-edges，构建全局词典并转为直方图
3. 依照 3-split averaging 规则计算逻辑级 Wasserstein-1 距离
4. (可选) 依据抽象公式距离进行层次聚类重新排序
5. 绘制热图

2025-06-14
"""
# from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import ot                         # pip install pot

# ───────────────────────────────────────────── regex & util constants ──
MASK_RE   = re.compile(r"(logic_\d{3})_split(\d+)_part([AB])\.json$")
ASSIGN_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s+is\s+(True|False)", re.I)
RULE_RE   = re.compile(r"\(\s*(.+?)\s*\)\s*->")


# ═════════════════════════════ 1. 基础工具 ═══════════════════════════════
def load_edges(path: Path) -> List[str]:
    """Flatten nested-list JSON → list[str] of edges with positive score."""
    raw = json.loads(path.read_text())
    edges: List[str] = []

    def rec(name: str, arr):
        if isinstance(arr, (int, float)):
            if arr > 0:
                edges.append(name)
        else:
            for i, v in enumerate(arr):
                rec(f"{name}:{i}", v)

    for mod, arr in raw.items():
        rec(mod, arr)
    return edges


# ═════════════════════════════ 2. 读入 & 组织数据 ═════════════════════════
def collect_bags(root_pattern: str, seed: int
                 ) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """data[logic][split][part] = list_of_edges"""
    if any(ch in root_pattern for ch in "*?[]"):
        files = sorted(Path().glob(root_pattern))
    else:
        root  = Path(root_pattern)
        files = sorted(root.rglob(f"logic_*_split{seed}_part*.json"))

    if not files:
        raise RuntimeError(f"No JSON files found under: {root_pattern}")

    data: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for f in files:
        if (m := MASK_RE.match(f.name)):
            logic, split, part = m.groups()
            data[logic][split][part] = load_edges(f)
    return data


# ═════════════════════════════ 3. OT 距离矩阵 ═════════════════════════════
def build_ot_matrix(data: Dict[str, Dict[str, Dict[str, List[str]]]]
                    ) -> tuple[np.ndarray, List[str]]:
    logics = sorted(data.keys())
    n      = len(logics)

    # —— 建全局词典 → 直方图 —— #
    vocab = {e: i for i, e in enumerate(
        sorted({e for lg in data.values()
                  for sp in lg.values()
                  for bag in sp.values()
                  for e in bag}))
    }
    m = len(vocab)
    if m == 0:
        raise RuntimeError("No positive edges in any file.")

    def to_hist(bag: List[str]) -> np.ndarray:
        vec = np.bincount([vocab[e] for e in bag], minlength=m).astype(float)
        return vec / max(1, vec.sum())

    H = {(lg, sp, pt): to_hist(bag)
         for lg, d_sp in data.items()
         for sp, d_pt in d_sp.items()
         for pt, bag in d_pt.items()}

    C = np.ones((m, m), dtype=float) - np.eye(m)        # ground-cost matrix

    # —— 逻辑级 OT 距离 —— #
    mat = np.zeros((n, n))
    for i, li in enumerate(logics):
        for j, lj in enumerate(logics):
            if i == j:
                dists = [
                    ot.emd2(H[(li, s, "A")], H[(li, s, "B")], C)
                    for s, parts in data[li].items()
                    if {"A", "B"} <= parts.keys()
                ]
                mat[i, j] = np.mean(dists) if dists else 0.0
            else:
                split_vals = []
                for s in set(data[li]) & set(data[lj]):
                    Ai = H.get((li, s, "A")); Bi = H.get((li, s, "B"))
                    Aj = H.get((lj, s, "A")); Bj = H.get((lj, s, "B"))
                    combos = []
                    if Ai is not None and Bj is not None: combos.append(ot.emd2(Ai, Bj, C))
                    if Bi is not None and Aj is not None: combos.append(ot.emd2(Bi, Aj, C))
                    if Ai is not None and Aj is not None: combos.append(ot.emd2(Ai, Aj, C))
                    if Bi is not None and Bj is not None: combos.append(ot.emd2(Bi, Bj, C))
                    if combos:
                        split_vals.append(np.mean(combos))
                mat[i, j] = np.mean(split_vals) if split_vals else 0.0
    return mat, logics


# ═══════════════════════════ 4. (可选) 抽象聚类排序 ═══════════════════════
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
    all_vars      = set(v1) | set(v2)
    truth_diff    = sum(v1.get(k) != v2.get(k) for k in all_vars if k in v1 and k in v2)
    missing_diff  = sum(1 for k in all_vars if (k in v1) ^ (k in v2))
    ant_num_diff  = abs(len(ant1) - len(ant2))
    op_diff       = 0 if op1 == op2 else 1
    return truth_diff + missing_diff + ant_num_diff + op_diff

def reorder_by_abstract(mat: np.ndarray, labels: List[str]
                        ) -> tuple[np.ndarray, List[str]]:
    abs_formulas = load_abstract_formulas()
    n = len(labels)
    abs_mat = np.zeros((n, n))
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            abs_mat[i, j] = abstract_distance(
                abs_formulas.get(li, ""), abs_formulas.get(lj, "")
            )
    order = sch.leaves_list(sch.linkage(abs_mat, method="average"))[::-1]
    return mat[order][:, order], [labels[i] for i in order]


# ═════════════════════════════ 5. 绘图 ═══════════════════════════════════
def plot_heatmap(mat: np.ndarray, labels: List[str], out_png: Path,
                 cmap: str = "magma", block: int = 3):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n*.5), max(5, n*.5)))
    im = ax.imshow(mat, cmap=cmap)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise Optimal-Transport Distance (3-split avg)")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"✓ Saved heat-map → {out_png}")


# ═════════════════════════════ 6. CLI ════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(
        OUTPUTS_DIR / "attr_scores/qwen1_5_1_8b_chat/0_1"),
        help="Directory or glob of *.json mask files")
    p.add_argument("--out_csv",
        help="Output CSV path for raw OT matrix "
             "(default: <computed_out_dir>/ot_distance_raw.csv)")
    p.add_argument("--block", type=int, default=3,
        help="Dashed grid every N cells")
    p.add_argument("--no_cluster", action="store_true",
        help="Skip abstract-distance reordering")
    p.add_argument("--seed", type=int, default=0,
        help="Seed (only used to find *_split<seed> files when --input is a directory)")
    p.add_argument("--resume", type=Path,
        help="Path to checkpoint; results go under a subdir named after the checkpoint stem")
    return p


def main():
    args = build_parser().parse_args()

    # --- Decide output directory (same pattern as other scripts) ---
    base_dir = Path(args.input).resolve()
    if args.resume:
        ckpt_stem = Path(args.resume).stem
        out_dir = base_dir / ckpt_stem
        print(f"[info] Using subdir based on checkpoint name: {out_dir}")
    else:
        out_dir = base_dir / "origin"  # or "unmodified"
        print(f"[info] Using subdir for unmodified model: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # File paths under out_dir (unless --out_csv explicitly given)
    out_png = out_dir / "ot_distance_matrix.png"
    out_csv = Path(args.out_csv) if args.out_csv else out_dir / "ot_distance_raw.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Compute + save ---
    data           = collect_bags(args.input, args.seed)
    ot_mat, labels = build_ot_matrix(data)

    pd.DataFrame(ot_mat, index=labels, columns=labels).to_csv(out_csv, float_format="%.6f")
    print("✓ Saved raw matrix →", out_csv)

    if not args.no_cluster:
        ot_mat, labels = reorder_by_abstract(ot_mat, labels)

    plot_heatmap(ot_mat, labels, out_png, block=args.block)


# ═════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
