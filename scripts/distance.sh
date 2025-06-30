cd /scratch/vjd5zr/project/ReasonEdit

CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.distance.pot_distance --input "results/output/attr_scores/qwen2_5_3b_instruct/5"
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.distance.jaccard --input "results/output/attr_scores/qwen2_5_3b_instruct/10"

