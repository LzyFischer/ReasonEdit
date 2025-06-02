
CUDA_VISIBLE_DEVICES=0 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --quest True > logs/tinyllama_quest.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --knowledge True --cot True > logs/tinyllama_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --knowledge True --fine_tune True > logs/llama_knowledge_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --reason True --cot True > logs/llama_reason_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --reason True --fine_tune True > logs/tinyllama_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --gt True --cot True > logs/llama_gt_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --gt True --fine_tune True > logs/tinyllama_gt_finetune.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python train.py --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --quest True > logs/llama_7b_quest.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --knowledge True --cot True > logs/llama_7b_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --knowledge True --fine_tune True > logs/llama_knowledge_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --reason True --cot True > logs/llama_reason_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --reason True --fine_tune True > logs/llama_7b_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --gt True --cot True > logs/llama_gt_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train.py --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --gt True --fine_tune True > logs/llama_7b_gt_finetune.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python train.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --lr 1e-4 --quest True > logs/llama_3b_quest.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --lr 1e-4 --knowledge True --cot True > logs/llama_3b_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --knowledge True --fine_tune True > logs/llama_knowledge_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --reason True --cot True > logs/llama_reason_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --lr 1e-4 --reason True --fine_tune True > logs/llama_3b_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --gt True --cot True > logs/llama_gt_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --lr 1e-4 --gt True --fine_tune True > logs/llama_3b_gt_finetune.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python train.py --model_name "openai-community/gpt2" --lr 1e-3 --quest True > logs/gpt_small_quest.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python train.py --model_name "openai-community/gpt2" --lr 1e-3 --knowledge True --cot True > logs/gpt_small_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --knowledge True --fine_tune True > logs/llama_knowledge_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --reason True --cot True > logs/llama_reason_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python train.py --model_name "openai-community/gpt2" --lr 1e-3 --reason True --fine_tune True > logs/gpt_small_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python train.py --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --gt True --cot True > logs/llama_gt_cot.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python train.py --model_name "openai-community/gpt2" --lr 1e-3 --gt True --fine_tune True > logs/gpt_small_gt_finetune.log 2>&1 &