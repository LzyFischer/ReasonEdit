#!/usr/bin/env python3
import torch
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.eap_yh.circuit_discovery import find_node_importance, get_random_circuit, find_sig_nodes
from src.eap_yh.circuit_ablation import CircuitAblator

# ---------- config ----------
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_NAME = "meta-llama/CodeLlama-7b-Instruct-hf"
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

TOY_DATA = [
    {
        "clean_text":    "A is True. If A then B. Is B True? Answer:",
        "corrupted_text":"A is False. If A then B. Is B True? Answer:",
        "answer": "true"
    },
    {
        "clean_text":    "X is False. If not X then Y. Is Y True? Answer:",
        "corrupted_text":"X is True.  If not X then Y. Is Y True? Answer:",
        "answer": "true"
    },
]


def main():
    # 1) load model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    # name_list = [name for name, param in mdl.named_parameters() if "model.layers.10" in name]
    # mdl.eval()

    # 2) discover node importances (uses EdgeAttributePatcher internally)
    nodes = find_node_importance(tok, mdl, TOY_DATA, DEVICE)
    
    # nodes: List[(node_name, effect_tensor, activation_tensor)]

    # 3) pick a tiny circuit (random or top-k). Example: random with 2 heads/MLPs per layer:
    # toy_circuit_nx = get_random_circuit(nodes, topk=2, max_layers=len(mdl.model.layers))
    toy_circuit_nx = find_sig_nodes(
        nodes, topk=2, max_layers=len(mdl.model.layers)
    )


    # 4) ablate & measure
    ablator = CircuitAblator(mdl, tok, DEVICE)

    # convert nx graph -> list of node strings expected by CircuitAblator
    mc_nodes = list(toy_circuit_nx.nodes)

    # minimal “dataset” for evaluation (same TOY_DATA, but CircuitAblator.measure_performance
    # expects [{"prompt": ..., "answer": ...}] — adapt to your fn signature!)
    eval_set = [{"prompt": d["clean_text"], "answer": d["answer"]} for d in TOY_DATA]

    # run analysis (you may need to match param names exactly to your file)
    results = ablator.analyze_functional_changes(
        mc_idx   = 0,
        mc_nodes = mc_nodes,
        dataset  = eval_set,
        func_interp = {}
    )

    print("Original acc:", results["original"]["accuracy"])
    print("Ablated  acc:", results["ablated"]["accuracy"])
    print("Δacc:", results["original"]["accuracy"] - results["ablated"]["accuracy"])

if __name__ == "__main__":
    main()