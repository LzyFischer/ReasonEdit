#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python circuit_aio.py --model gpt2-large
CUDA_VISIBLE_DEVICES=1 python circuit_aio.py --model qwen1.5-0.5b-chat