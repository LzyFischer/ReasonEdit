#!/bin/bash

python -m src.main \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --deductive_json "data/logic/deductive_logic.json" \
    --augmented_json "data/corrupt/augmented_dataset.json"