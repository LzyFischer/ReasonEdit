from __future__ import annotations

import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.reasonedit_plus import ReasonEditPlus, set_seed
import pdb


# ───────────────────────────── 1.  Helpers ──────────────────────────────
def greedy_answer(prompt: str, tok: AutoTokenizer, model, device) -> str:
    """Greedy-decode one answer token after ‘Answer:’ and return lowercase word."""
    with torch.no_grad():
        ids = tok(prompt + " Answer:", return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=8, do_sample=False)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return txt.split("Answer:")[-1].strip().split()[0].lower()


def build_pairs(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Group `rows` by logic and output (pair list, logic type list)."""
    bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bucket[r["logic"]].append(r)

    pairs = []
    for lg, lg_rows in bucket.items():
        for r in lg_rows:
            pairs.append(
                {
                    "logic": lg,
                    "train": {"text": r["prompt"], "label": r["gold"]},
                    "eval":  {"prompt": r["prompt"], "gold": r["gold"]},
                }
            )
    return pairs, sorted(bucket.keys())


# ───────────────────────────── 2.  CLI ──────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ReasonEdit+ per-logic sweep")
    p.add_argument("--model_name",      default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--deductive_json", type=Path,
                   default=Path("data/logic/deductive_logic.json"))
    p.add_argument("--augmented_json", type=Path,
                   default=Path("data/corrupt/augmented_dataset.json"))
    p.add_argument("--out_dir",        type=Path, default=Path("output/perlogic"))

    # phase hyper-params
    p.add_argument("--ce_steps",   type=int, default=2)
    p.add_argument("--lora_steps", type=int, default=30)

    # rng
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ───────────────────────── 3.  Main routine ─────────────────────────────
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # shared ReasonEdit+ instance (one model for all pairs)
    editor_args = argparse.Namespace(
        **vars(args),
        ce_lr=2e-8, lora_lr=1e-4, lambda_ce=1.0, tau=1e-4,
        lora_rank=8, lora_alpha=32, lora_dropout=0.05,
        lora_target_modules="q_proj,k_proj,v_proj,o_proj", 
        batch_size=1
    )
    editor      = ReasonEditPlus(editor_args)
    tok, model  = editor.tok, editor.model
    device      = editor.device

    # build evaluation pairs
    pairs, logic_types = build_pairs(editor.logic_rows)
    idx_of = {lg: i for i, lg in enumerate(logic_types)}
    n_logic = len(logic_types)

    # accuracy tracking
    mat_total   = [[0] * n_logic for _ in range(n_logic)]
    mat_correct = [[0] * n_logic for _ in range(n_logic)]
    hits = total = 0

    dev_set = random.sample(editor.logic_rows, k=50)
    dev_prompts = [
        {"prompt": r["prompt"] + " (Answer in True, False, or N/A (Neither)). Answer:", "gold": r["gold"], "logic": r["logic"]}
        for r in dev_set
    ]

    # # iterate over pairs
    for step, p in enumerate(tqdm(pairs, desc="pairs"), 1):
        # Phase I: contrastive enhancement (uses editor.aug_rows internally)
        logic = p["logic"]

    #     dev_hits = 0
    #     dev_total = 0
    #     for dev in tqdm(dev_prompts):
    #         pred = greedy_answer(dev["prompt"], tok, model, device)
    #         if dev["gold"] in pred:
    #             dev_hits += 1
    #         dev_total += 1
    #     dev_acc = dev_hits / dev_total
    #     print(f"[Eval Post Logic-{logic}] Dev Accuracy: {dev_acc:.3f}")

        editor.contrastive_enhance(
            sample_same = (lambda lg=logic: editor.aug_bucket[logic]),
            sample_diff = (lambda lg=logic: [
                r for k, rows in editor.aug_bucket.items() if k != logic for r in rows
            ]),
        )

        dev_hits = 0
        dev_total = 0
        for dev in tqdm(dev_prompts):
            pred = greedy_answer(dev["prompt"], tok, model, device)
            if dev["gold"] in pred:
                dev_hits += 1
            dev_total += 1
        dev_acc = dev_hits / dev_total
        print(f"[Eval Post Logic-{logic}] Dev Accuracy: {dev_acc:.3f}")

        # Phase II: LoRA local edit on the current (clean, label)
        pdb.set_trace()
        editor.local_lora_edit_single(
            {"clean": p["train"]["text"], "answer": p["train"]["label"]}
        )

        # evaluate
        row = idx_of[p["logic"]]
        pred = greedy_answer(p["eval"]["prompt"], tok, model, device)
        correct = p["eval"]["gold"] in pred

        mat_total[row][row] += 1
        total += 1
        if correct:
            mat_correct[row][row] += 1
            hits += 1

        if step % 10 == 0 or step == len(pairs):
            print(f"[{step}/{len(pairs)}]  running_acc = {hits / total:.3f}")

    # save per-logic accuracy matrix
    acc = [
        [
            mat_correct[r][c] / mat_total[r][c] if mat_total[r][c] else 0.0
            for c in range(n_logic)
        ]
        for r in range(n_logic)
    ]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(acc, index=logic_types, columns=logic_types).to_csv(
        out / "accuracy.csv", float_format="%.4f"
    )
    print("✓ accuracy matrix saved to", out)


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()