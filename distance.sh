# CUDA_VISIBLE_DEVICES=0 python edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/1"
# CUDA_VISIBLE_DEVICES=1 python edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/0_1"
# CUDA_VISIBLE_DEVICES=2 python edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/0_5"

# CUDA_VISIBLE_DEVICES=3 python pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/1"
# CUDA_VISIBLE_DEVICES=0 python pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/0_1"
# CUDA_VISIBLE_DEVICES=1 python pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/0_5"

# CUDA_VISIBLE_DEVICES=2 python jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/1"
# CUDA_VISIBLE_DEVICES=3 python jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/0_1"
# CUDA_VISIBLE_DEVICES=0 python jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_1_8b_chat/0_5"



# CUDA_VISIBLE_DEVICES=1 python edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/1"
# CUDA_VISIBLE_DEVICES=2 python edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/0_1"
# CUDA_VISIBLE_DEVICES=3 python edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/0_5"

# CUDA_VISIBLE_DEVICES=0 python pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/1"
# CUDA_VISIBLE_DEVICES=1 python pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/0_1"
# CUDA_VISIBLE_DEVICES=2 python pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/0_5"

# CUDA_VISIBLE_DEVICES=3 python jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/1"
# CUDA_VISIBLE_DEVICES=0 python jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/0_1"
# CUDA_VISIBLE_DEVICES=1 python jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen1_5_0_5b_chat/0_5"



CUDA_VISIBLE_DEVICES=0 python distance/edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/1"
CUDA_VISIBLE_DEVICES=0 python distance/edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/10"
CUDA_VISIBLE_DEVICES=0 python distance/edit_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/5"

CUDA_VISIBLE_DEVICES=1 python distance/pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/1"
CUDA_VISIBLE_DEVICES=0 python distance/pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/10"
CUDA_VISIBLE_DEVICES=0 python distance/pot_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/5"

CUDA_VISIBLE_DEVICES=0 python distance/jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/1"
CUDA_VISIBLE_DEVICES=0 python distance/jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/10"
CUDA_VISIBLE_DEVICES=0 python distance/jaccard_distance.py --input "/scratch/vjd5zr/project/ReasonEdit/output/attr_scores/qwen2_5_3b_instruct/5"

