#!/usr/bin/env python3
"""
批量跑 seed × distance × lr；全部用 `python -m pkg.module …` 方式调用
把所有“设置项”改为命令行参数，可灵活指定。
"""
from __future__ import annotations
import subprocess, itertools, argparse, os
from pathlib import Path
from glob import glob
import pdb  

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----------------------------- helpers -----------------------------
DISTANCE_MODULES_DEFAULT = {
    "pot": "src.preliminary.distance.pot_distance",
    "edit": "src.preliminary.distance.edit_distance",
    "jaccard": "src.preliminary.distance.jaccard_distance",
}

REQ_PNGS_DEFAULT = ["scatter.png", "scatter_binned.png",
                    "scatter_sliding.png", "scatter_acc.png"]

def run(cmd, **kw):
    print("➤", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_distance_mapping(kv_list: list[str]) -> dict[str, str]:
    """
    支持 --distance-map pot=src.preliminary.distance.pot_distance 形式的覆盖/新增
    """
    mapping = DISTANCE_MODULES_DEFAULT.copy()
    for kv in kv_list or []:
        if "=" not in kv:
            raise ValueError(f"--distance-map expects key=module, got: {kv}")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Bad mapping: {kv}")
        mapping[k] = v
    return mapping

def slug_lr(lr: float | str) -> str:
    s = str(lr)
    return s.replace(".", "p").replace("-", "m")

# ----------------------------- main -----------------------------
def main(args: argparse.Namespace):
    # Resolve distance mapping and subset to use
    distance_map = parse_distance_mapping(args.distance_map)
    if args.distances is None or len(args.distances) == 0:
        use_distances = list(distance_map.keys())
    else:
        unknown = [d for d in args.distances if d not in distance_map]
        if unknown:
            raise ValueError(f"Unknown distances: {unknown}. "
                             f"Known: {sorted(distance_map.keys())}")
        use_distances = args.distances

    resume = args.resume if (args.resume and str(args.resume).strip()) else None
    resume_tag = Path(resume).stem if resume is not None else "origin"

    # 1) circuits (默认关闭，与原脚本保持一致)
    if args.run_circuits:
        for seed in args.seeds:
            pattern = str(PROJECT_ROOT / f"results/output/attr_scores/{args.model_id}/10/{resume_tag}/seed{seed}_split*.json")
            matching_dirs = glob(pattern)
            if args.force or not matching_dirs:
                cmd = ["python", "-m", "src.preliminary.circuit.circuit_aio",
                       "--seed", str(seed),
                       "--out_root", str(PROJECT_ROOT / "results/output/attr_scores")]
                if resume is not None:
                    cmd += ["--resume", str(resume)]
                run(cmd)

    # 2) distances (默认关闭，与原脚本保持一致)
    if args.run_distances:
        for seed in args.seeds:
            for dist_name in use_distances:
                module = distance_map[dist_name]
                csv_path = PROJECT_ROOT / f"results/output/distance/{dist_name}/{resume_tag}/seed{seed}.csv"
                if args.force or not csv_path.exists():
                    input_dir = (PROJECT_ROOT /
                                 f"results/output/attr_scores/{args.model_id}/10/{resume_tag}")
                    ensure_dir(csv_path.parent)
                    cmd = ["python", "-m", module,
                           "--seed", str(seed),
                           "--input", str(input_dir),
                           "--out_csv", str(csv_path)]
                    if resume is not None:
                        cmd += ["--resume", str(resume)]
                    run(cmd)

    # 3) per-logic Δ-accuracy（原脚本启用）
    if args.run_perlogic:
        for lr in args.lrs:
            lr_slug = slug_lr(lr)
            csv_root = PROJECT_ROOT / f"results/output/perlogic/{lr_slug}/{resume_tag}"
            if args.force or not csv_root.exists():
                ensure_dir(csv_root)
                cmd = ["python", "-m", "src.preliminary.edit.perlogic_delta_batch",
                       "--lr", str(lr),
                       "--out_root", str(PROJECT_ROOT / "results/output/perlogic")]
                if resume is not None:
                    cmd += ["--resume", str(resume)]
                run(cmd)

    # 4) combined plots
    if args.plot_mode in {"combined", "both"}:
        for dist_name in use_distances:
            for lr in args.lrs:
                seeds_str = ",".join(str(s) for s in args.seeds)
                cmd = [
                    "python", "-m", "tools.plot_preliminary_combined",
                    "--distance", dist_name,
                    "--seeds", seeds_str,
                    "--lrs", str(lr),
                    "--combine", str(args.combine_size or 0),
                ]
                if resume is not None:
                    cmd += ["--resume", str(resume)]
                run(cmd)

    # 4-B) per-run plots
    if args.plot_mode in {"single", "both"}:
        req_pngs = args.req_pngs or REQ_PNGS_DEFAULT
        for seed, dist_name, lr in itertools.product(args.seeds, use_distances, args.lrs):
            lr_slug = slug_lr(lr)
            out_dir = PROJECT_ROOT / f"results/figures/{dist_name}/seed{seed}/lr{lr_slug}/{resume_tag}"
            need = args.force or not out_dir.exists() or not all((out_dir / f).exists() for f in req_pngs)
            if need:
                ensure_dir(out_dir)
                cmd = [
                    "python", "-m", "tools.plot_preliminary_intra_inter",
                    "--distance", dist_name,
                    "--seed", str(seed),
                    "--lr", str(lr),
                ]
                if resume is not None:
                    cmd += ["--resume", str(resume)]
                run(cmd, cwd=out_dir)
        
    if args.plot_mode in {"cluster"}:
        for dist_name in use_distances:
            cmd = [
                "python", "-m", "tools.plot_cluster",
                "--distance", dist_name,
                "--lrs", " ".join(str(lr) for lr in args.lrs),
                "--seeds", " ".join(str(s) for s in args.seeds),
                "--combine", str(args.combine_size or 0),
            ]
            if resume is not None:
                cmd += ["--subdir", str(resume_tag)]
            run(cmd)

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="批量跑 seed × distance × lr；全部用 `python -m pkg.module …` 方式调用（参数化版本）"
    )
    # core sweep params
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                    help="要运行的随机种子，如：--seeds 0 1 2")
    ap.add_argument("--model-id", default="qwen2_5_3b_instruct",
                    help="模型 ID（影响 attr_scores 输入路径）")
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-5],
                    help="学习率列表，如：--lrs 1e-5 2e-5")
    ap.add_argument("--distances", nargs="*", default=None,
                    help=f"使用哪些距离指标名（默认全用）：{','.join(DISTANCE_MODULES_DEFAULT.keys())}")
    ap.add_argument("--distance-map", nargs="*", default=None,
                    help=("覆盖/新增距离名到模块的映射，形如："
                          "pot=src.preliminary.distance.pot_distance "
                          "edit=src.preliminary.distance.edit_distance"))
    ap.add_argument("--req-pngs", nargs="*", default=REQ_PNGS_DEFAULT,
                    help="单次绘图需要存在的 PNG 列表用于跳过（single 模式下）。")

    # plotting & combine
    ap.add_argument("--plot-mode", choices=["single", "combined", "both", "cluster"],
                    default="cluster",
                    help="绘图模式：single/combined/both")
    ap.add_argument("--combine-size", type=int, default=10,
                    help="combined 模式合并的 run 数量；0 或 None 表示合并全部。")

    # resume & control
    ap.add_argument("--resume", default=None,
                    help="可选的 checkpoint 路径；留空表示 origin。")
    ap.add_argument("--force", action="store_true",
                    help="强制重跑/重绘，忽略已有输出。")

    def str2bool(v: str) -> bool:
        """将字符串转换为布尔值，支持 'true', 'false', '1', '0' 等。"""
        return v.lower() in {'true', '1', 'yes'}

    # stage toggles（与原脚本行为对齐：circuits/distances 默认关闭，perlogic 默认开启）
    ap.add_argument("--run-circuits", default=False, type=str2bool,
                    help="执行 circuits 计算步骤（默认开启）。")
    ap.add_argument("--run-distances", default=False, type=str2bool,
                    help="执行 distances 计算步骤（默认开启）。")
    ap.add_argument("--run-perlogic", default=True, type=str2bool,
                    help="执行 per-logic Δ-accuracy 计算（默认开启）。")
    args = ap.parse_args()

    # 若用户显式传了 --no-run-perlogic（可通过环境变量或移除默认），可在此扩展；保持简单：默认 True
    main(args)