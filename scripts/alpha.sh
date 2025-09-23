# python -m src.alpha_edit --src_json data/logic/level_2_1.json --lr 2e-5 --seed 42 > logs/level_3_1/alpha_seed42_2e-5.log 2>&1

# python -m src.alpha_edit --src_json data/logic/level_1.json --lr 5e-5 --seed 42 > logs/level_1/alpha_seed42_5e-5.log 2>&1

# python -m src.alpha_edit --src_json data/logic/level_2_1.json --lr 2e-5 --seed 0 > logs/level_2_1/alpha_seed0_2e-5.log 2>&1

# python -m src.alpha_edit --src_json data/logic/level_1.json --lr 2e-5 --seed 42 > logs/level_1/alpha_seed42_2e-5.log 2>&1


python -m src.alpha_edit_math --src_json data/logic/math.json --lr 1e-4 --seed 15 > logs/math/alpha_seed15_1e-4.log 2>&1
python -m src.alpha_edit_math --src_json data/logic/math.json --lr 1e-4 --seed 16 > logs/math/alpha_seed16_1e-4.log 2>&1
python -m src.alpha_edit_math --src_json data/logic/math.json --lr 1e-4 --seed 17 > logs/math/alpha_seed17_1e-4.log 2>&1

python -m src.alpha_edit_math --src_json data/logic/math.json --lr 2e-4 --seed 15 > logs/math/alpha_seed15_2e-4.log 2>&1
python -m src.alpha_edit_math --src_json data/logic/math.json --lr 2e-4 --seed 16 > logs/math/alpha_seed16_2e-4.log 2>&1
python -m src.alpha_edit_math --src_json data/logic/math.json --lr 2e-4 --seed 17 > logs/math/alpha_seed17_2e-4.log 2>&1

