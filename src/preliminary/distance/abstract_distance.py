#!/usr/bin/env python3
"""
abstract_distance_matrix.py
────────────────────────────────────────────────────────────────────────────
Usage
─────
python abstract_distance_matrix.py \
       --formulas data/formulas.json \
       --out_dir  results/

• `--formulas`  points to either
     – a JSON file that maps e.g.  {"logic_000": "<formula str>", ...}
     – **or** a directory that contains text files named   logic_000.txt …
• `--out_dir`   where the CSV will be written (default = alongside formulas file)

Distance definition
───────────────────
truth value differences + missing-variable penalties
+ antecedent-length difference + operator mismatch
exactly as in your original script.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ASSIGN_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s+is\s+(True|False)", re.I)
RULE_RE   = re.compile(r"\(\s*(.+?)\s*\)\s*->")

# ───────────────────────────── formula parsing ────────────────────────────
def parse_formula(text: str) -> Tuple[Dict[str,bool], set[str], str]:
    """Return (var_truths, antecedent_vars, operator)."""
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
    return var_vals, ant_vars, op or "none"

def abstract_distance(f1: str, f2: str) -> int:
    v1, ant1, op1 = parse_formula(f1)
    v2, ant2, op2 = parse_formula(f2)
    all_vars     = set(v1) | set(v2)
    truth_diff   = sum(v1.get(k) != v2.get(k) for k in all_vars if k in v1 and k in v2)
    missing_diff = sum(1 for k in all_vars if (k in v1) ^ (k in v2))
    ant_num_diff = abs(len(ant1) - len(ant2))
    op_diff      = 0 if op1 == op2 else 1
    return truth_diff + missing_diff + ant_num_diff + op_diff

# ──────────────────────────── I/O helpers ─────────────────────────────────
def read_formulas(path: Path) -> Dict[str,str]:
    if path.is_file():
        with path.open() as f:
            return json.load(f)
    # directory of *.txt
    mapping = {}
    for p in sorted(path.glob("logic_*.txt")):
        mapping[p.stem] = p.read_text()
    if not mapping:
        raise RuntimeError(f"No formulas found under {path}")
    return mapping

def build_matrix(mapping: Dict[str,str]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted(mapping.keys())
    n    = len(keys)
    mat  = np.zeros((n,n), dtype=int)
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i <= j:                         # symmetric → half compute
                d = abstract_distance(mapping[k1], mapping[k2])
                mat[i,j] = mat[j,i] = d
    return mat, keys

# ─────────────────────────────────── CLI ──────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--formulas",default="data/logic/abstract.json",
                   help="JSON file OR directory containing logic_XXX.txt files")
    p.add_argument("--out_dir", default="results/output/distance",
                   help="Where to write CSV (default = same directory as --formulas)")
    p.add_argument("--resume", type=Path,
                help="Path to checkpoint (expects {'model_state': ...} or raw state_dict)")
    return p.parse_args()

def main() -> None:
    args      = get_args()
    path      = Path(args.formulas).expanduser()
    out_dir   = Path(args.out_dir) if args.out_dir else path.parent

    if args.resume:
        ckpt_stem = Path(args.resume).stem
        out_dir = out_dir / ckpt_stem
        print(f"[info] Using subdir based on checkpoint name: {out_dir}")
    else:
        out_dir = out_dir / "origin"  # or "unmodified"
        print(f"[info] Using subdir for unmodified model: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)


    formulas  = read_formulas(path)
    mat, lbls = build_matrix(formulas)

    csv_path  = out_dir / "abstract_distance_matrix.csv"
    pd.DataFrame(mat, index=lbls, columns=lbls).to_csv(csv_path)
    print(f"✓ Saved abstract-distance matrix → {csv_path}")

if __name__ == "__main__":
    main()
