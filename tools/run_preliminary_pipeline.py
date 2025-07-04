#!/usr/bin/env python3
"""
批量跑 seed × distance × lr；全部用 `python -m pkg.module …` 方式调用
"""
import subprocess, itertools, argparse
from pathlib import Path
import pdb
from glob import glob

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEEDS      = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29,]
MODEL_ID    = "qwen2_5_3b_instruct"
DISTANCES  = {"pot": "src.preliminary.distance.pot_distance",
              "edit": "src.preliminary.distance.edit_distance",
              "jaccard": "src.preliminary.distance.jaccard_distance"}
LRS        = [1e-4, 1.5e-4]
REQ_PNGS    = ["scatter.png", "scatter_binned.png",
               "scatter_sliding.png", "scatter_acc.png"]
plot_mode = "combined"  # "single", "combined", "both"
combine_size = 9  # None or 0 → merge ALL runs
len_pattern = 10


def run(cmd, **kw):
    print("➤", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)

def main(force: bool = False):
    # 1) circuits
    for seed in SEEDS:
        # 检查是否已有该 split 的结果（例如 logic_XXX_split{seed}_part 目录）
        pattern = str(PROJECT_ROOT / f"results/output/attr_scores/{MODEL_ID}/10/logic_*_split{seed}_part*")
        matching_dirs = glob(pattern)

        if force or not matching_dirs:
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
    if plot_mode in {"combined", "both"}:
        for dist_name in DISTANCES.keys():
            for lr in LRS:
                seeds_str = ",".join(str(s) for s in SEEDS)
                lr_slug = str(lr).replace('.', 'p')
                run([
                    "python", "-m", "tools.plot_preliminary_combined",
                    "--distance", dist_name,
                    "--seeds", seeds_str,
                    "--lrs",   str(lr),
                    "--combine", str(combine_size or 0)  # 0 → merge ALL runs
                ])

    # 4-B) original per-run plots
    if plot_mode in {"single", "both"}:
        for seed, (dist_name, _), lr in itertools.product(SEEDS, DISTANCES.items(), LRS):
            lr_slug = str(lr).replace('.', 'p')
            out_dir = PROJECT_ROOT / f"results/figures/{dist_name}/seed{seed}/lr{lr_slug}"
            need    = force or not out_dir.exists() or not all((out_dir / f).exists() for f in REQ_PNGS)

            if need:
                run([
                    "python", "-m", "tools.plot_preliminary",
                    "--distance", dist_name,
                    "--seed", str(seed),
                    "--lr",   str(lr),
                ])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
