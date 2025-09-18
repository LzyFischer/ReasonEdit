#!/bin/bash
set -euo pipefail
# mkdir -p logs/lora_edit

python -m src.lora_edit --src_json data/logic/level_1.json --lr 2e-4 --seed 1 > logs/level_1/l1_lr2e-4_s1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_2_1.json --lr 2e-4 --seed 1 > logs/level_2_1/l2_lr2e-4_s1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 2e-4 --seed 1 > logs/level_3_1/l3_lr2e-4_s1.log 2>&1
