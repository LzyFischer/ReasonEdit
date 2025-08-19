#!/bin/bash

# 运行第一个命令并输出到 log1.txt
CUDA_VISIBLE_DEVICES=4 python -um tools.run_preliminary_pipeline --lrs 5e-5 --resume "ckpts/reptile_ns_00045.pt" > tmp/nc5.txt 2>&1 &

# 运行第二个命令并输出到 log2.txt
CUDA_VISIBLE_DEVICES=1 python -um tools.run_preliminary_pipeline --lrs 5e-5 --resume "ckpts/contrastive_00045.pt" > tmp/cc5.txt 2>&1 &


CUDA_VISIBLE_DEVICES=5 python -um src.lora_edit --lr 3.5e-5 --resume ckpts/reptile_00045.pt > tmp/35_r.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 python -um src.lora_edit --lr 4e-5 --resume ckpts/reptile_ns_00045.pt > tmp/4_n.txt 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -um src.lora_edit --lr 4e-5 --resume ckpts/contrastive_00045.pt > tmp/4_c.txt 2>&1 &
# 等待两个进程结束
wait