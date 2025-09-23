
# ──────────────── seed = 5 ────────────────
# # level_3_1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 8e-6 --seed 5 > logs/lora_edit/l3_base_lr8e-6_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 5 > logs/lora_edit/l3_nometa_lr1.5e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 5 > logs/lora_edit/l3_reptiled_lr1.5e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 5 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s5_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 8e-6 --seed 5 > logs/lora_edit/l3_reptilensmnl_lr8e-6_s5_gpu1.log 2>&1

# # level_2_1
python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 8e-6 --seed 5 > logs/lora_edit/l2_base_lr8e-6_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 1e-4 --seed 5 > logs/lora_edit/l2_nometa_lr1e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 1e-4 --seed 5 > logs/lora_edit/l2_reptiled_lr1e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 1e-4 --seed 5 > logs/lora_edit/l2_reptilensm_lr1e-4_s5_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 8e-6 --seed 5 > logs/lora_edit/l2_reptilensmnl_lr8e-6_s5_gpu1.log 2>&1

# level_1
python -m src.lora_edit --src_json data/logic/level_1.json --lr 8e-6 --seed 5 > logs/lora_edit/l1_base_lr8e-6_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 1e-4 --seed 5 > logs/lora_edit/l1_nometa_lr1e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 1e-4 --seed 5 > logs/lora_edit/l1_reptiled_lr1e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 1e-4 --seed 5 > logs/lora_edit/l1_reptilensm_lr1e-4_s5_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 8e-6 --seed 5 > logs/lora_edit/l1_reptilensd_lr8e-6_s5_gpu1.log 2>&1



# ──────────────── seed = 5 ────────────────
# # level_3_1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 5e-6 --seed 5 > logs/lora_edit/l3_base_lr5e-6_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 5 > logs/lora_edit/l3_nometa_lr1.5e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 5 > logs/lora_edit/l3_reptiled_lr1.5e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 5 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s5_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 5e-6 --seed 5 > logs/lora_edit/l3_reptilensmnl_lr5e-6_s5_gpu1.log 2>&1

# # level_2_1
python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 5e-6 --seed 5 > logs/lora_edit/l2_base_lr5e-6_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 1e-4 --seed 5 > logs/lora_edit/l2_nometa_lr1e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 1e-4 --seed 5 > logs/lora_edit/l2_reptiled_lr1e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 1e-4 --seed 5 > logs/lora_edit/l2_reptilensm_lr1e-4_s5_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 5e-6 --seed 5 > logs/lora_edit/l2_reptilensmnl_lr5e-6_s5_gpu1.log 2>&1

# level_1
python -m src.lora_edit --src_json data/logic/level_1.json --lr 5e-6 --seed 5 > logs/lora_edit/l1_base_lr5e-6_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 1e-4 --seed 5 > logs/lora_edit/l1_nometa_lr1e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 1e-4 --seed 5 > logs/lora_edit/l1_reptiled_lr1e-4_s5_gpu1.log 2>&1
# CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 1e-4 --seed 5 > logs/lora_edit/l1_reptilensm_lr1e-4_s5_gpu1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 5e-6 --seed 5 > logs/lora_edit/l1_reptilensd_lr5e-6_s5_gpu1.log 2>&1