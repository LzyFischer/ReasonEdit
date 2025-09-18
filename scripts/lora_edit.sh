# #!/bin/bash
# set -euo pipefail
# mkdir -p logs/lora_edit

# # ──────────────── 示例（原文中是注释，已加好重定向；若要运行请去掉行首 #） ────────────────
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1e-4 > logs/lora_edit/l3_nometa_lr1e-4.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1e-4 > logs/lora_edit/l3_reptiled_lr1e-4.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1e-4 > logs/lora_edit/l3_reptilensm_lr1e-4.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 1e-4 > logs/lora_edit/l3_reptilensmnl_lr1e-4.log 2>&1


# # ──────────────── seed = 42 ────────────────
# # # level_3_1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 1.5e-4 --seed 42 > logs/lora_edit/l3_base_lr1.5e-4_s42.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 42 > logs/lora_edit/l3_nometa_lr1.5e-4_s42.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 42 > logs/lora_edit/l3_reptiled_lr1.5e-4_s42.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 42 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s42.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 1.5e-4 --seed 42 > logs/lora_edit/l3_reptilensmnl_lr1.5e-4_s42.log 2>&1

# # # level_2_1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 1e-4 --seed 42 > logs/lora_edit/l2_base_lr1e-4_s42.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 1e-4 --seed 42 > logs/lora_edit/l2_nometa_lr1e-4_s42.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 1e-4 --seed 42 > logs/lora_edit/l2_reptiled_lr1e-4_s42.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 1e-4 --seed 42 > logs/lora_edit/l2_reptilensm_lr1e-4_s42.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 1e-4 --seed 42 > logs/lora_edit/l2_reptilensmnl_lr1e-4_s42.log 2>&1

# # level_1
# # python -m src.lora_edit --src_json data/logic/level_1.json --lr 1e-4 --seed 42 > logs/lora_edit/l1_base_lr1e-4_s42.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 1e-4 --seed 42 > logs/lora_edit/l1_nometa_lr1e-4_s42.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 1e-4 --seed 42 > logs/lora_edit/l1_reptiled_lr1e-4_s42.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 1e-4 --seed 42 > logs/lora_edit/l1_reptilensm_lr1e-4_s42.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 1e-4 --seed 42 > logs/lora_edit/l1_reptilensd_lr1e-4_s42.log 2>&1


# # ──────────────── seed = 1 ────────────────
# # # level_3_1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 1.5e-4 --seed 1 > logs/lora_edit/l3_base_lr1.5e-4_s1.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 1 > logs/lora_edit/l3_nometa_lr1.5e-4_s1.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 1 > logs/lora_edit/l3_reptiled_lr1.5e-4_s1.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 1 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s1.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 1.5e-4 --seed 1 > logs/lora_edit/l3_reptilensmnl_lr1.5e-4_s1.log 2>&1

# # # level_2_1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 1e-4 --seed 1 > logs/lora_edit/l2_base_lr1e-4_s1.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 1e-4 --seed 1 > logs/lora_edit/l2_nometa_lr1e-4_s1.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 1e-4 --seed 1 > logs/lora_edit/l2_reptiled_lr1e-4_s1.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 1e-4 --seed 1 > logs/lora_edit/l2_reptilensm_lr1e-4_s1.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 1e-4 --seed 1 > logs/lora_edit/l2_reptilensmnl_lr1e-4_s1.log 2>&1

# # level_1
# # python -m src.lora_edit --src_json data/logic/level_1.json --lr 1e-4 --seed 1 > logs/lora_edit/l1_base_lr1e-4_s1.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 1e-4 --seed 1 > logs/lora_edit/l1_nometa_lr1e-4_s1.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 1e-4 --seed 1 > logs/lora_edit/l1_reptiled_lr1e-4_s1.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 1e-4 --seed 1 > logs/lora_edit/l1_reptilensm_lr1e-4_s1.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 1e-4 --seed 1 > logs/lora_edit/l1_reptilensd_lr1e-4_s1.log 2>&1


# # ──────────────── seed = 24 ────────────────
# # # level_3_1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 1.5e-4 --seed 24 > logs/lora_edit/l3_base_lr1.5e-4_s24.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 24 > logs/lora_edit/l3_nometa_lr1.5e-4_s24.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 24 > logs/lora_edit/l3_reptiled_lr1.5e-4_s24.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 24 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s24.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 1.5e-4 --seed 24 > logs/lora_edit/l3_reptilensmnl_lr1.5e-4_s24.log 2>&1

# # # level_2_1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 1e-4 --seed 24 > logs/lora_edit/l2_base_lr1e-4_s24.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 1e-4 --seed 24 > logs/lora_edit/l2_nometa_lr1e-4_s24.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 1e-4 --seed 24 > logs/lora_edit/l2_reptiled_lr1e-4_s24.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 1e-4 --seed 24 > logs/lora_edit/l2_reptilensm_lr1e-4_s24.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 1e-4 --seed 24 > logs/lora_edit/l2_reptilensmnl_lr1e-4_s24.log 2>&1

# # level_1
# # python -m src.lora_edit --src_json data/logic/level_1.json --lr 1e-4 --seed 24 > logs/lora_edit/l1_base_lr1e-4_s24.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 1e-4 --seed 24 > logs/lora_edit/l1_nometa_lr1e-4_s24.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 1e-4 --seed 24 > logs/lora_edit/l1_reptiled_lr1e-4_s24.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 1e-4 --seed 24 > logs/lora_edit/l1_reptilensm_lr1e-4_s24.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 1e-4 --seed 24 > logs/lora_edit/l1_reptilensd_lr1e-4_s24.log 2>&1


# # ──────────────── seed = 114514 ────────────────
# # # level_3_1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 1.5e-4 --seed 114514 > logs/lora_edit/l3_base_lr1.5e-4_s114514.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1.5e-4 --seed 114514 > logs/lora_edit/l3_nometa_lr1.5e-4_s114514.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1.5e-4 --seed 114514 > logs/lora_edit/l3_reptiled_lr1.5e-4_s114514.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1.5e-4 --seed 114514 > logs/lora_edit/l3_reptilensm_lr1.5e-4_s114514.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 1.5e-4 --seed 114514 > logs/lora_edit/l3_reptilensmnl_lr1.5e-4_s114514.log 2>&1

# # # level_2_1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 1e-4 --seed 114514 > logs/lora_edit/l2_base_lr1e-4_s114514.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/nometa_00045.pt --lr 1e-4 --seed 114514 > logs/lora_edit/l2_nometa_lr1e-4_s114514.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_d_00045.pt --lr 1e-4 --seed 114514 > logs/lora_edit/l2_reptiled_lr1e-4_s114514.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 1e-4 --seed 114514 > logs/lora_edit/l2_reptilensm_lr1e-4_s114514.log 2>&1
# # python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 1e-4 --seed 114514 > logs/lora_edit/l2_reptilensmnl_lr1e-4_s114514.log 2>&1

# # level_1
# # python -m src.lora_edit --src_json data/logic/level_1.json --lr 1e-4 --seed 114514 > logs/lora_edit/l1_base_lr1e-4_s114514.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/nometa_00045.pt --lr 1e-4 --seed 114514 > logs/lora_edit/l1_nometa_lr1e-4_s114514.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_d_00040.pt --lr 1e-4 --seed 114514 > logs/lora_edit/l1_reptiled_lr1e-4_s114514.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsm_00040.pt --lr 1e-4 --seed 114514 > logs/lora_edit/l1_reptilensm_lr1e-4_s114514.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 1e-4 --seed 114514 > logs/lora_edit/l1_reptilensd_lr1e-4_s114514.log 2>&1

python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 2e-4 --seed 1 > logs/level_1/l1_reptilensd_lr1e-4_s1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 2e-4 --seed 1 > logs/level_2_1/l2_reptilensmnl_lr1e-4_s1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 2e-4 --seed 1 > logs/level_3_1/l3_reptilensmnl_lr1e-4_s1.log 2>&1
