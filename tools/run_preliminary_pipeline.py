#!/usr/bin/env python3
"""
批量跑 seed × distance × lr；全部用 `python -m pkg.module …` 方式调用
"""
import subprocess, itertools, argparse
from pathlib import Path
import pdb
from glob import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEEDS      = [0, 1, 2, 3, 4, 7, 8, 9, 11, 16, 6, 43, 44, 45]
MODEL_ID    = "qwen2_5_3b_instruct"
DISTANCES  = {"pot": "src.preliminary.distance.pot_distance",
              "edit": "src.preliminary.distance.edit_distance",
              "jaccard": "src.preliminary.distance.jaccard_distance"}
LRS        = [1.5e-4]
REQ_PNGS    = ["scatter.png", "scatter_binned.png",
               "scatter_sliding.png", "scatter_acc.png"]


def run(cmd, **kw):
    print("➤", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)

def main(force: bool = False):
    # 1) circuits
    for seed in SEEDS:
        # 检查是否已有该 split 的结果（例如 logic_XXX_split{seed}_part 目录）
        pattern = str(PROJECT_ROOT / f"results/output/attr_scores/{MODEL_ID}/10/logic_*_split{seed}_part*")
        matching_dirs = glob(pattern)

        if force or len(matching_dirs) == 0:
            run(["python", "-m", "src.preliminary.circuit.circuit_aio",
                 "--seed", str(seed)])

    # 2) distances
    for seed in SEEDS:
        for name, module in DISTANCES.items():
            csv_path = PROJECT_ROOT / f"results/output/distance/{name}/seed{seed}.csv"
            if force or not csv_path.exists():
                input_dir = (PROJECT_ROOT /
                            f"results/output/attr_scores/{MODEL_ID}/10")
                run(["python", "-m", module, 
                     "--seed", str(seed),
                    "--input", str(input_dir),
                    "--out_csv", str(csv_path)])
   

    # 3) per-logic Δ-accuracy
    for lr in LRS:
        lr_slug = str(lr).replace(".", "p").replace("-", "m")
        csv_root = PROJECT_ROOT / f"results/output/perlogic/{lr_slug}"
        if force or not csv_root.exists():
            run(["python", "-m", "src.preliminary.edit.perlogic_delta",
                 "--lr", str(lr)])

    # 4) 绘图

    for seed, (dist_name, _), lr in itertools.product(SEEDS, DISTANCES.items(), LRS):
        lr_slug   = str(lr).replace('.', 'p')
        out_dir   = PROJECT_ROOT / f"results/figures/{dist_name}/seed{seed}/lr{lr_slug}"
        need_rerun = force or not out_dir.exists() or not all(
            (out_dir / fname).exists() for fname in REQ_PNGS
        )

        if need_rerun:
            run(
                [
                    "python", "-m", "tools.plot_preliminary",
                    "--distance", dist_name,
                    "--seed", str(seed),
                    "--lr",   str(lr),
                ]
            )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
