# aab is True.\naaa is True.\n(aaa or aab) -> aac.\nDeduce the result of aac."
# aaa is True.\naab is False.\n(aaa or aab) -> aac.\nDeduce the result of aac.
# aab is False.\naaa is False.\n(aaa and aab) -> aac.\nDeduce the result of aac.
# aab is True.\naaa is True.\n(aaa and aab) -> aac.\nDeduce the result of aac.
# aaa is False.\naab is True.\n(aaa and aab) -> aac.\nDeduce the result of aac.
# aaa is True.\naab is False.\n(aaa and aab) -> aac.\nDeduce the result of aac.




# python -m src.reptile_ns_dist_gen --data_json "data/corrupt/level_3_1.json" --include_rows 1,2
# python -m src.reptile_ns_dist_gen --data_json "data/corrupt/level_3_1.json" --include_rows 1,2,3,4
# python -m src.reptile_ns_dist_gen --data_json "data/corrupt/level_3_1.json" --include_rows 1,2,3,4,5,6
# python -m src.reptile_ns_dist_gen --data_json "data/corrupt/level_3_1.json" --include_rows 1,2,3,4,5,6,7,8



python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 3e-5 --resume "ckpts/level_3_1/reptile_nsmnl_logic2.pt" --seed 17 > logs/level_3_1/lora_reptile_nsmnl_logic2_seed17_3e-5.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 3e-5 --resume "ckpts/level_3_1/reptile_nsmnl_logic2.pt" --seed 20 > logs/level_3_1/lora_reptile_nsmnl_logic2_seed20_3e-5.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 4e-5 --resume "ckpts/level_3_1/reptile_nsmnl_logic2.pt" --seed 17 > logs/level_3_1/lora_reptile_nsmnl_logic2_seed17_4e-5.log 2>&1
python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 4e-5 --resume "ckpts/level_3_1/reptile_nsmnl_logic2.pt" --seed 20 > logs/level_3_1/lora_reptile_nsmnl_logic2_seed20_4e-5.log 2>&1

# python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 3e-5 --resume "ckpts/level_3_1/reptile_nsmnl_logic4.pt" --seed 17 > logs/level_3_1/lora_reptile_nsmnl_logic4_seed17_3e-5.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 3e-5 --resume "ckpts/level_3_1/reptile_nsmnl_logic6.pt" --seed 17 > logs/level_3_1/lora_reptile_nsmnl_logic6_seed17_3e-5.log 2>&1
# python -m src.lora_edit --src_json data/logic/level_3_1.json --lr 3e-5 --resume "ckpts/level_3_1/reptile_nsmnl_logic8.pt" --seed 17 > logs/level_3_1/lora_reptile_nsmnl_logic8_seed17_3e-5.log 2>&1


