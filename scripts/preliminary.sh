
# # CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --quest True > results/logs/preliminary/tinyllama_quest.log 2>&1 &
# # CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/tinyllama_knowledge_cot.log 2>&1 &
# # # CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --knowledge True --fine_tune True > results/logs/preliminary/llama_knowledge_finetune.log 2>&1 &
# # # CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --reason True --cot True > results/logs/preliminary/llama_reason_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 9e-5 --reason True --fine_tune True > results/logs/preliminary/tinyllama_reason_finetune.log 2>&1 &
# # # CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 1e-4 --gt True --cot True > results/logs/preliminary/llama_gt_cot.log 2>&1 &
# # CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --gt True --fine_tune True > results/logs/preliminary/tinyllama_gt_finetune.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --quest True > results/logs/preliminary/llama_7b_quest.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/llama_7b_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --knowledge True --fine_tune True > results/logs/preliminary/llama_knowledge_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --reason True --cot True > results/logs/preliminary/llama_reason_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 9e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --gt True --cot True > results/logs/preliminary/llama_gt_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 2e-4 --gt True --fine_tune True > results/logs/preliminary/llama_7b_gt_finetune.log 2>&1 &


# # CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B-Instruct" --lr 1e-4 --quest True > results/logs/preliminary/llama_3b_quest.log 2>&1 &
# # CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B-Instruct" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/llama_3b_knowledge_cot.log 2>&1 &
# # # CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --knowledge True --fine_tune True > results/logs/preliminary/llama_knowledge_finetune.log 2>&1 &
# # # CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --reason True --cot True > results/logs/preliminary/llama_reason_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B-Instruct" --lr 9e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_reason_finetune.log 2>&1 &
# # # CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --gt True --cot True > results/logs/preliminary/llama_gt_cot.log 2>&1 &
# # CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B-Instruct" --lr 2e-4 --gt True --fine_tune True > results/logs/preliminary/llama_3b_gt_finetune.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2" --lr 7e-4 --quest True > results/logs/preliminary/gpt_small_quest.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2" --lr 7e-4 --knowledge True --cot True > results/logs/preliminary/gpt_small_knowledge_cot.log 2>&1 &
# # CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --knowledge True --fine_tune True > results/logs/preliminary/llama_knowledge_finetune.log 2>&1 &
# # CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --reason True --cot True > results/logs/preliminary/llama_reason_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2" --lr 65e-5 --reason True --fine_tune True > results/logs/preliminary/gpt_small_reason_finetune.log 2>&1 &
# # # CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --lr 2e-4 --gt True --cot True > results/logs/preliminary/llama_gt_cot.log 2>&1 &
# # CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2" --lr 8e-4 --gt True --fine_tune True > results/logs/preliminary/gpt_small_gt_finetune.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2" --lr 1e-4 --quest True > results/logs/preliminary/gpt2_quest.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/gpt2_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2" --lr 1e-4 --reason True --fine_tune True > results/logs/preliminary/gpt2_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2" --lr 1e-4 --gt True --fine_tune True > results/logs/preliminary/gpt2_gt_finetune.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2-large" --lr 1e-4 --quest True > results/logs/preliminary/gpt2_large_quest.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2-large" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/gpt2_large_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2-large" --lr 1e-4 --reason True --fine_tune True > results/logs/preliminary/gpt2_large_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "openai-community/gpt2-large" --lr 1e-4 --gt True --fine_tune True > results/logs/preliminary/gpt2_large_gt_finetune.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen1.5-1.8B" --lr 1e-4 --quest True > results/logs/preliminary/qwen1_8_quest.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen1.5-1.8B" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/qwen1_8_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen1.5-1.8B" --lr 1e-4 --reason True --fine_tune True > results/logs/preliminary/qwen1_8_1e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen1.5-1.8B" --lr 1e-4 --gt True --fine_tune True > results/logs/preliminary/qwen1_8_gt_finetune.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen1.5-0.5B" --lr 1e-4 --quest True > results/logs/preliminary/qwen0_5_quest.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen1.5-0.5B" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/qwen0_5_knowledge_cot.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen1.5-0.5B" --lr 1e-4 --reason True --fine_tune True > results/logs/preliminary/qwen0_5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen1.5-0.5B" --lr 1e-4 --gt True --fine_tune True > results/logs/preliminary/qwen0_5_gt_finetune.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1e-4 --reason True --fine_tune True > results/logs/preliminary/qwen3_it_1e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 2e-4 --reason True --fine_tune True > results/logs/preliminary/qwen3_it_2e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1.5e-4 --reason True --fine_tune True > results/logs/preliminary/qwen3_it_15e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 5e-5 --reason True --fine_tune True > results/logs/preliminary/qwen3_it_5e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 5e-6 --reason True --fine_tune True > results/logs/preliminary/qwen3_it_5e6_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1e-4 --reason True --fine_tune True > results/logs/preliminary/qwen3_it_1e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "gemma-2b-it" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/qwen3_knowledge_cot.log 2>&1 &



# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 1e-4 --reason True --fine_tune True > results/logs/preliminary/llama_3b_1e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 2e-4 --reason True --fine_tune True > results/logs/preliminary/llama_3b_2e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 5e-4 --reason True --fine_tune True > results/logs/preliminary/llama_3b_5e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 5e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_5e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 2e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_2e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/llama_3b_knowledge_cot.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --reason True --fine_tune True > results/logs/preliminary/llama_7b_1e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 2e-4 --reason True --fine_tune True > results/logs/preliminary/llama_7b_2e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 5e-4 --reason True --fine_tune True > results/logs/preliminary/llama_7b_5e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 5e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_5e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 2e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_2e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/llama_7b_knowledge_cot.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-7B-Instruct" --lr 1.6e-4 --reason True --fine_tune True > results/logs/preliminary/qwen25_7b_16e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-7B-Instruct" --lr 1.7e-4 --reason True --fine_tune True > results/logs/preliminary/qwen25_7b_17e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-7B-Instruct" --lr 1.8e-4 --reason True --fine_tune True > results/logs/preliminary/qwen25_7b_18e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-7B-Instruct" --lr 1.9e-4 --reason True --fine_tune True > results/logs/preliminary/qwen25_7b_19e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-7B-Instruct" --lr 5e-4 --reason True --fine_tune True > results/logs/preliminary/qwen25_7b_5e4_reason_finetune.log 2>&1 &
# # CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-7B-Instruct" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/qwen25_7b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-7B-Instruct" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/qwen25_7b_knowledge_cot.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "google/gemma-2b-it" --lr 3e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_2b_3e5_reason_finetune.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "google/gemma-2b-it" --lr 4e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_2b_4e5_reason_finetune.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "google/gemma-2b-it" --lr 1.5e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_2b_9e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "google/gemma-2b-it" --lr 5e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_2b_5e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "google/gemma-2b-it" --lr 2e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_2b_2e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "google/gemma-2b-it" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_2b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "google/gemma-2b-it" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/gemma_2b_knowledge_cot.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 python -m src.preliminary.edit.edit --model_name "google/gemma-7b-it" --lr 6.5e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_7b_65e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "google/gemma-7b-it" --lr 8e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_7b_8e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "google/gemma-7b-it" --lr 9e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_7b_9e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "google/gemma-7b-it" --lr 3e-4 --reason True --fine_tune True > results/logs/preliminary/gemma_7b_3e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "google/gemma-7b-it" --lr 2e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_7b_2e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "google/gemma-7b-it" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/gemma_7b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "google/gemma-7b-it" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/gemma_7b_knowledge_cot.log 2>&1 &




# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 3e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_3e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 6e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_6e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 8e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_8e5_reason_finetune.log 2>&1 
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 5e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_5e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 2e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_2e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/llama_7b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-2-7b-chat-hf" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/llama_7b_knowledge_cot.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 3e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_3e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 6e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_6e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 8e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_8e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 5e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_5e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 2e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_2e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/llama_3b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "meta-llama/Llama-3.2-3B" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/llama_3b_knowledge_cot.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2-7B-Instruct" --lr 1.3e-4 --reason True --fine_tune True > results/logs/preliminary/qwen2_7b_13e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2-7B-Instruct" --lr 1.5e-4 --reason True --fine_tune True > results/logs/preliminary/qwen2_7b_15e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2-7B-Instruct" --lr 1.7e-4 --reason True --fine_tune True > results/logs/preliminary/qwen2_7b_17e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2-7B-Instruct" --lr 5e-5 --reason True --fine_tune True > results/logs/preliminary/qwen2_7b_5e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2-7B-Instruct" --lr 2e-5 --reason True --fine_tune True > results/logs/preliminary/qwen2_7b_2e5_reason_finetune.log 2>&1 
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2-7B-Instruct" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/qwen2_7b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2-7B-Instruct" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/qwen2_7b_knowledge_cot.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1.6e-4 --reason True --fine_tune True > results/logs/preliminary/qwen25_3b_16e4_reason_finetune.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1.7e-4 --reason True --fine_tune True > results/logs/preliminary/qwen25_3b_17e4_reason_finetune.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1.8e-4 --reason True --fine_tune True > results/logs/preliminary/qwen25_3b_18e4_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 5e-5 --reason True --fine_tune True > results/logs/preliminary/qwen25_3b_5e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 2e-5 --reason True --fine_tune True > results/logs/preliminary/qwen25_3b_2e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1e-5 --reason True --fine_tune True > results/logs/preliminary/qwen25_3b_1e5_reason_finetune.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -m src.preliminary.edit.edit --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1e-4 --knowledge True --cot True > results/logs/preliminary/qwen25_3b_knowledge_cot.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python preliminary_perlogic.py --model_name "Qwen/Qwen2.5-3B-Instruct" --lr 1.5e-4 --reason True --fine_tune True > results/logs/preliminary_perlogic/qwen25_3b_15e4_reason_finetune.log 2>&1 &
