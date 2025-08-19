#!/bin/bash

# # 运行第一个命令并输出到 log1.txt
# python -um src.lora_edit --lr 0 > tmp/0_o.txt 2>&1 &

# # 运行第二个命令并输出到 log2.txt
# python -um src.lora_edit --lr 5e-5 > tmp/5_o.txt 2>&1

# # 运行第一个命令并输出到 log1.txt
# python -um src.lora_edit --lr 0 --resume ckpts/reptile_00045.pt > tmp/0_r.txt 2>&1 &

# 运行第二个命令并输出到 log2.txt
python -um src.lora_edit --lr 4e-5 --resume ckpts/reptile_00045.pt > tmp/4_r.txt 2>&1
python -um src.lora_edit --lr 4e-5 --resume ckpts/reptile_ns_00045.pt > tmp/4_n.txt 2>&1
python -um src.lora_edit --lr 4e-5 --resume ckpts/contrastive_00045.pt > tmp/4_c.txt 2>&1

# # 运行第一个命令并输出到 log1.txt
# python -um src.lora_edit --lr 0 --resume ckpts/reptile_00045.pt > tmp/0_r.txt 2>&1 &

# # 运行第二个命令并输出到 log2.txt
python -um src.lora_edit --lr 2e-5 --resume ckpts/reptile_00045.pt > tmp/2_r.txt 2>&1
# 等待两个进程结束
wait