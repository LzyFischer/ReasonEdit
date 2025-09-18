

python -m src.ns_dist --data_json data/corrupt/level_2_1.json --seed 42
CUDA_VISIBLE_DEVICES=1 python -m src.reptile_dist --data_json data/corrupt/level_2_1.json --seed 42

python -m src.reptile_ns_dist --data_json data/corrupt/level_2_1.json --seed 42
CUDA_VISIBLE_DEVICES=1 python -m src.reptile_ns --data_json data/corrupt/level_2_1.json --seed 42

python -m src.ns_dist --data_json data/corrupt/level_3_1.json --seed 42
CUDA_VISIBLE_DEVICES=1 python -m src.reptile_dist --data_json data/corrupt/level_3_1.json --seed 42

python -m src.reptile_ns_dist --data_json data/corrupt/level_3_1.json --seed 42
CUDA_VISIBLE_DEVICES=1 python -m src.reptile_ns --data_json data/corrupt/level_3_1.json --seed 42




# python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00040.pt --lr 2e-4

python -m src.lora_edit --src_json data/logic/level_2_1.json --resume ckpts/level_2_1/reptile_nsm_00045.pt --lr 1e-4



python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/nometa_00045.pt --lr 1e-4

CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_d_00045.pt --lr 1e-4

python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsm_00045.pt --lr 1e-4

CUDA_VISIBLE_DEVICES=1 python -m src.lora_edit --src_json data/logic/level_3_1.json --resume ckpts/level_3_1/reptile_nsmnl_00045.pt --lr 1e-4



python -m tools.run_preliminary_pipeline --plot-mode cluster --resume ckpts/level_1/reptile_nsm_00040.pt
python -m tools.run_preliminary_pipeline --plot-mode cluster --resume ckpts/level_1/nometa_00045.pt
python -m tools.run_preliminary_pipeline --plot-mode cluster --resume ckpts/level_1/reptile_d_00040.pt
python -m tools.run_preliminary_pipeline --plot-mode cluster --resume ckpts/level_1/reptile_nsd_00045.pt
python -m tools.run_preliminary_pipeline --plot-mode cluster 