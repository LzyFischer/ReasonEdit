# CUDA_VISIBLE_DEVICES=0 python preliminary_fc.py --model_name "google/gemma-2b-it" > logs/preliminary_fc/gemma_2b_o.log 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=0 python preliminary_fc.py --model_name "Qwen/Qwen1.5-7B-Chat" > logs/preliminary_fc/qwen15_7bchat.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python preliminary_fc.py --model_name "Qwen/Qwen2.5-7B" > logs/preliminary_fc/qwen25_7b_o.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python preliminary_fc.py --model_name "Qwen/Qwen-7B-Chat" > logs/preliminary_fc/qwen_7bchat.log 2>&1 &

# sleep 20
# CUDA_VISIBLE_DEVICES=2 python preliminary_fc.py --model_name "meta-llama/Llama-3.2-3B" > logs/preliminary_fc/llama_3b.log 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=3 python preliminary_fc.py --model_name "meta-llama/Llama-2-7b-chat-hf" > logs/preliminary_fc/llama_7b.log 2>&1 &
# sleep 20
CUDA_VISIBLE_DEVICES=0 python preliminary_fc.py --model_name "Qwen/Qwen2.5-3B-Instruct" > logs/preliminary_fc/qwen_3b.log 2>&1 &
# sleep 20
# CUDA_VISIBLE_DEVICES=2 python preliminary_fc.py --model_name "google/gemma-7b-it" > logs/preliminary_fc/gemma_7b.log 2>&1 &


