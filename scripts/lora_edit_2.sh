#!/bin/bash
set -euo pipefail
mkdir -p logs/lora_edit

# ──────────────── seed = 0（第 1 遍） ────────────────
# # level_3_1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 5e-4 --seed 42 > logs/lora_edit/l3_base_lr5e-4_s42_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 0 > logs/lora_edit/l3_nometa_lr1.5e-4_s0_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 0 > logs/lora_edit/l3_reptiled_lr1.5e-4_s0_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 0 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s0_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 5e-4 --seed 42 > logs/lora_edit/l3_reptilensmnl_lr5e-4_s42_gpu1.log 2>&1

# # level_2_1
python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 5e-4 --seed 42 > logs/lora_edit/l2_base_lr5e-4_s42_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 5e-4 --seed 0 > logs/lora_edit/l2_nometa_lr5e-4_s0_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 5e-4 --seed 0 > logs/lora_edit/l2_reptiled_lr5e-4_s0_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 5e-4 --seed 0 > logs/lora_edit/l2_reptilensm_lr5e-4_s0_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 5e-4 --seed 42 > logs/lora_edit/l2_reptilensmnl_lr5e-4_s42_gpu1.log 2>&1

# level_1
python -m src.lora_edit --src_json data/logic/level_1.json --lr 5e-4 --seed 42 > logs/lora_edit/l1_base_lr5e-4_s42_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 5e-4 --seed 0 > logs/lora_edit/l1_nometa_lr5e-4_s0_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 5e-4 --seed 0 > logs/lora_edit/l1_reptiled_lr5e-4_s0_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 5e-4 --seed 0 > logs/lora_edit/l1_reptilensm_lr5e-4_s0_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 5e-4 --seed 42 > logs/lora_edit/l1_reptilensd_lr5e-4_s42_gpu1.log 2>&1


# ──────────────── seed = 0（第 2 遍，避免覆盖加 _b 后缀） ────────────────
# # level_3_1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 3e-4 --seed 1 > logs/lora_edit/l3_base_lr3e-4_s1_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 0 > logs/lora_edit/l3_nometa_lr1.5e-4_s0_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 0 > logs/lora_edit/l3_reptiled_lr1.5e-4_s0_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 0 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s0_gpu1_b.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 3e-4 --seed 1 > logs/lora_edit/l3_reptilensmnl_lr3e-4_s1_gpu1_b.log 2>&1

# # level_2_1
python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 3e-4 --seed 1 > logs/lora_edit/l2_base_lr3e-4_s1_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 5e-4 --seed 0 > logs/lora_edit/l2_nometa_lr5e-4_s0_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 5e-4 --seed 0 > logs/lora_edit/l2_reptiled_lr5e-4_s0_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 5e-4 --seed 0 > logs/lora_edit/l2_reptilensm_lr5e-4_s0_gpu1_b.log 2>&1
python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 3e-4 --seed 1 > logs/lora_edit/l2_reptilensmnl_lr3e-4_s1_gpu1_b.log 2>&1

# level_1
python -m src.lora_edit --src_json data/logic/level_1.json --lr 3e-4 --seed 1 > logs/lora_edit/l1_base_lr3e-4_s1_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 5e-4 --seed 0 > logs/lora_edit/l1_nometa_lr5e-4_s0_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 5e-4 --seed 0 > logs/lora_edit/l1_reptiled_lr5e-4_s0_gpu1_b.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 5e-4 --seed 0 > logs/lora_edit/l1_reptilensm_lr5e-4_s0_gpu1_b.log 2>&1
python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 3e-4 --seed 1 > logs/lora_edit/l1_reptilensd_lr3e-4_s1_gpu1_b.log 2>&1


# ──────────────── seed = 101010 ────────────────
# # level_3_1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 7e-4 --seed 114514 > logs/lora_edit/l3_base_lr7e-4_s114514_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 101010 > logs/lora_edit/l3_nometa_lr1.5e-4_s101010_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 101010 > logs/lora_edit/l3_reptiled_lr1.5e-4_s101010_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 101010 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s101010_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 7e-4 --seed 114514 > logs/lora_edit/l3_reptilensmnl_lr7e-4_s114514_gpu1.log 2>&1

# # level_2_1
python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 7e-4 --seed 114514 > logs/lora_edit/l2_base_lr7e-4_s114514_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 5e-4 --seed 101010 > logs/lora_edit/l2_nometa_lr5e-4_s101010_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 5e-4 --seed 101010 > logs/lora_edit/l2_reptiled_lr5e-4_s101010_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 5e-4 --seed 101010 > logs/lora_edit/l2_reptilensm_lr5e-4_s101010_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 7e-4 --seed 114514 > logs/lora_edit/l2_reptilensmnl_lr7e-4_s114514_gpu1.log 2>&1

# level_1
python -m src.lora_edit --src_json data/logic/level_1.json --lr 7e-4 --seed 114514 > logs/lora_edit/l1_base_lr7e-4_s114514_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 5e-4 --seed 101010 > logs/lora_edit/l1_nometa_lr5e-4_s101010_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 5e-4 --seed 101010 > logs/lora_edit/l1_reptiled_lr5e-4_s101010_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 5e-4 --seed 101010 > logs/lora_edit/l1_reptilensm_lr5e-4_s101010_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 7e-4 --seed 114514 > logs/lora_edit/l1_reptilensd_lr7e-4_s114514_gpu1.log 2>&1

