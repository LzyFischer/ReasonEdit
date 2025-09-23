#!/bin/bash
set -euo pipefail
# mkdir -p logs/lora_edit

python -m src.lora_edit_math --src_json data/logic/leve.json --lr 2e-4 --seed 1 > logs/math/lora_lr2e-4_s1.log 2>&1
python -m src.lora_edit_math --src_json data/logic/math.json --lr 2e-4 --seed 2 > logs/math/lora_lr2e-4_s2.log 2>&1
python -m src.lora_edit_math --src_json data/logic/math.json --lr 2e-4 --seed 3 > logs/math/lora_lr2e-4_s3.log 2>&1
python -m src.lora_edit_math --src_json data/logic/math.json --lr 1e-4 --seed 1 > logs/math/lora_lr1e-4_s1.log 2>&1
python -m src.lora_edit_math --src_json data/logic/math.json --lr 1e-4 --seed 2 > logs/math/lora_lr1e-4_s2.log 2>&1
python -m src.lora_edit_math --src_json data/logic/math.json --lr 1e-4 --seed 3 > logs/math/lora_lr1e-4_s3.log 2>&1
python -m src.lora_edit_math --src_json data/logic/math.json --lr 1.5e-4 --seed 1 > logs/math/lora_lr15e-4_s1.log 2>&1
python -m src.lora_edit_math --src_json data/logic/math.json --lr 1.5e-4 --seed 2 > logs/math/lora_lr15e-4_s2.log 2>&1
python -m src.lora_edit_math --src_json data/logic/math.json --lr 1.5e-4 --seed 3 > logs/math/lora_lr15e-4_s3.log 2>&1
