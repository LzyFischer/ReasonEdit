python -m src.lora_edit_bio --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/step_400_bio --lr 2e-4 > logs/level_3_1/l3_400_bio_lr2e-4.log 2>&1
python -m src.lora_edit_bio --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/step_400_bio --lr 2e-4  > logs/level_2_1/l2_400_bio_lr2e-4.log 2>&1
python -m src.lora_edit_bio --src_json data/logic/level_1.json --resume ckpts/level_1/step_400_bio --lr 2e-4 > logs/level_1/l1_400_bio_lr2e-4.log 2>&1


python -m src.rome_edit --rome_alpha 10.0 --src_json ./data/logic/level_2_1.json --edit_layer -10 --seed 17 > logs/level_2_1/rome_alpha10_layer-10_seed17.log 2>&1
python -m src.rome_edit --rome_alpha 10.0 --src_json ./data/logic/level_3_1.json --edit_layer -10 --seed 17 > logs/level_2_1/rome_alpha10_layer-10_seed17.log 2>&1
python -m src.rome_edit --rome_alpha 10.0 --src_json ./data/logic/level_1.json --edit_layer -10 --seed 17 > logs/level_2_1/rome_alpha10_layer-10_seed17.log 2>&1


python -m src.alpha_edit --src_json data/logic/level_1.json --lr 1e-4 --seed 42 > logs/level_1/alpha_seed42_1e-4.log 2>&1
python -m src.alpha_edit --src_json data/logic/level_2_1.json --lr 1e-4 --seed 42 > logs/level_2_1/alpha_seed42_1e-4.log 2>&1
python -m src.alpha_edit --src_json data/logic/level_3_1.json --lr 1e-4 --seed 42 > logs/level_3_1/alpha_seed42_1e-4.log 2>&1


python -m src.lora_edit --src_json data/logic/level_1.json --resume ckpts/level_1/reptile_nsd_00045.pt --lr 2e-4 --seed 1 > logs/level_1/l1_reptilensd_lr1e-4_s1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsmnl_00045.pt --lr 2e-4 --seed 1 > logs/level_2_1/l2_reptilensmnl_lr1e-4_s1.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 2e-4 --seed 1 > logs/level_3_1/l3_reptilensmnl_lr1e-4_s1.log 2>&1


