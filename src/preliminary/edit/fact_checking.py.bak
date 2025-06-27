#!/usr/bin/env python
"""
Sequentially fine‐tune a LoRA adapter on **one (train, eval) pair at a time**
and immediately evaluate on the corresponding full‐question prompt.
After each pair, the script prints the running accuracy over all processed pairs.

Changes vs. the buggy version provided by the user  ⤵
• No deepcopy — the same LoRA‐augmented model is updated throughout.
• One global optimiser tied to the LoRA parameters.
• Removed all `copy` / `sacrebleu` clutter.
• Fixed device handling and prompt/label masking.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb  # noqa: F401
from tqdm import tqdm
from collections import defaultdict
import difflib, string, re

###############################################################################
# 0. Reproducibility, args & paths
###############################################################################
SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser(
    description="Sequential LoRA fine‐tuning on logic pairs"
)


def str2bool(v):
    """Convert a string to a boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes"}:
        return True
    elif v.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


parser.add_argument(
    "--quest", type=str2bool, default=False, help="Use question‐answer pairs"
)
parser.add_argument(
    "--knowledge", type=str2bool, default=False, help="Use question‐answer pairs"
)
parser.add_argument("--cot", type=str2bool, default=False, help="Use CoT mode")
parser.add_argument(
    "--fine_tune", type=str2bool, default=False, help="Use fine‐tuning mode"
)
parser.add_argument("--reason", type=str2bool, default=False, help="Use reasoning mode")
parser.add_argument("--gt", type=str2bool, default=False, help="Use ground truth mode")
parser.add_argument(
    "--src_json",
    type=Path,
    default=Path("data/logic/deductive_logic.json"),
    help="Path to source JSON file",
)
parser.add_argument(
    "--lr",
    type=float,
    default=2e-4,
    help="Learning rate for the optimizer",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="meta-llama/Llama-3.2-3B",
    help="Name of the base model to use",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SRC_JSON = args.src_json


###############################################################################
# 1. Tokenizer & helpers
###############################################################################
MODEL_NAME = args.model_name
MAX_LEN = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def encode(row: dict):
    """
    Self-supervised: model sees `row["text"]` and must reproduce it.
    Only pad tokens are masked out of the loss.
    """
    ids = tokenizer(
        row["text"],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = ids["input_ids"].squeeze(0)
    attn_mask = ids["attention_mask"].squeeze(0)

    labels = input_ids.clone()
    labels[attn_mask == 0] = -100          # ignore loss on padding

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def generate_answer(prompt: str, model, max_new_tokens: int = 50) -> str:
    model.eval()
    with torch.no_grad():
        if "Answer:" not in prompt:
            templ = f"{prompt} (Answer in True, False, or N/A). Answer:"
        else:
            templ = f"{prompt}"
        ids = tokenizer(templ, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        try:
            answer = text.split(" Answer:")[-1].strip().split()[0].lower()
            if "true" not in answer and "false" not in answer and "n/a" not in answer:
                # extract the "not" as false, else as true after the "### Answer:"
                answers = text.split(" Answer:")[-1].lower()
                if "yes" in answers or "true" in answers:
                    answer = "true"
                elif "not" in answers or "no" in answers or "false" in answers:
                    answer = "false"
                elif "n/a" in answers:
                    answer = "n/a"
                else:
                    answer = "true"

        except IndexError:
            answer = ""
    return answer

###############################################################################
# 2. Model + LoRA adapter
###############################################################################
print("\nLoading base model (4‐bit)…")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # load_in_4bit=True,
    torch_dtype=torch.float16,
)

if args.model_name.startswith("google/gemma-3-4b-it"):
    target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
else:
    target_modules = None

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

model = get_peft_model(base_model, lora_cfg).to(device)
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


###############################################################################
# 3. Build (train, eval) pairs
###############################################################################


CUE = re.compile(
    r"\b(?:then|which implies|this (?:would )?implies?|would suggest that|"
    r"implies?|suggests? that)\b",
    re.I,
)


def split_premise(nl: str):
    """Return (antecedent, consequent) if the sentence has a clear split."""
    first, *rest = nl.split("\n", 2)
    premise = (rest[0] if rest else first).strip()
    if CUE.search(premise):
        ante, cons = CUE.split(premise, 1)
        return ante.strip(", ; "), CUE.search(premise).group(0) + cons.strip()

    sents = re.split(r"(?<=\.|!|\?)\s+", premise)
    return (
        (" ".join(sents[:-1]).strip(), sents[-1].strip())
        if len(sents) > 1
        else (None, None)
    )


###############################################################################
# 3. Build (train, eval) triplets   ⟵  REPLACES the old harvest_pairs()
###############################################################################
import re

TRUTH_PAT = re.compile(r"\b([a-z]{3})\s+is\s+(True|False)\b", re.I)

def extract_truths(question_nl: str) -> dict[str, str]:
    """
    从 'aaa is True.' 这种句子里抓出 { 'aaa': 'true', 'aab': 'false', ... }
    """
    table = {}
    for sym, val in TRUTH_PAT.findall(question_nl):
        table[f"<{sym.lower()}>"] = val.lower()   # ⇒ '<aaa>': 'true'
    return table

def harvest_triplets(path: Path):
    triplets = []
    root = json.load(path.open())

    for rec in root:
        gold_reason = str(rec["answer"]).lower()          # logic-question label

        # walk every domain/topic
        for domain in filter(lambda k: k not in {"question", "answer"}, rec):
            for topic, payload in rec[domain].items():
                aaa_txt = payload.get("<aaa>")
                aab_txt = payload.get("<aab>")            # may be premise or consequent
                aac_txt = payload.get("<aac>")            # may be missing
                nl_full = payload["<nl>"]

                # —— 先拿到这条记录的真值表 ————————————————
                truths = extract_truths(rec["question"][0]["<nl>"])

                # —— 生成 fact-check 列表 ——————————————————
                facts = []
                if aaa_txt and "<aaa>" in truths:
                    facts.append(("aaa", aaa_txt, truths["<aaa>"]))

                # 仍旧仅当有 <aac> 时才检查 <aab>
                if aac_txt and aab_txt and "<aab>" in truths:
                    facts.append(("aab", aab_txt, truths["<aab>"]))


                # ── evaluation tasks -----------------------------------------
                evals = [
                    {
                        "name": f"fact-{tag}",
                        "prompt": f"Is the following statement true or false? {txt}. (Answer in True or False). Answer:",
                        "gold": lbl,
                    }
                    for tag, txt, lbl in facts
                ] + [
                    {
                        "name": "reason",
                        "prompt": f"{nl_full}",
                        "gold": gold_reason,
                    }
                ]

                nl_lines = [l.rstrip() for l in nl_full.strip().split("\n") if l.strip()]
                premise_only = "\n".join(nl_lines[:-1]) if nl_lines[-1].endswith("?") else nl_full.strip()

                triplets.append(
                    {
                        "train": {"text": premise_only},    # ← just the premise text
                        "eval":  evals,                     # facts + reasoning checks
                    }
                )
    return triplets


pairs = harvest_triplets(SRC_JSON)
print(f"Loaded {len(pairs)} training examples.")

###############################################################################
# 4. Sequential fine-tuning + multi-task eval
###############################################################################
print("\nStarting sequential training…")
acc_correct, acc_total = defaultdict(int), defaultdict(int)

LORA_PAT = re.compile(r"lora")
orig_lora_state = {
    name: p.detach().clone()
    for name, p in model.named_parameters()
    if LORA_PAT.search(name)
}
print(f"Cached {len(orig_lora_state)} LoRA tensors.")

for idx, item in enumerate(pairs, 1):
    # # (a) reset LoRA weights
    # with torch.no_grad():
    #     for n, p in model.named_parameters():
    #         if n in orig_lora_state:
    #             p.copy_(orig_lora_state[n])
    # optimizer.zero_grad(set_to_none=True)

    # # (b) inner fine-tune (10 steps)
    # batch = encode(item["train"]); batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}
    # for _ in range(10):
    #     loss = model(**batch).loss
    #     loss.backward(); optimizer.step(); optimizer.zero_grad()

    # (c) evaluate all subtasks for this example
    for task in item["eval"]:
        # pred = generate_answer(task["prompt"], model)
        context_prompt = f"{item['train']['text']}\n\n{task['prompt']}"
        pred = generate_answer(context_prompt, model)
        acc_total[task["name"]] += 1
        if task["gold"].lower() in pred.lower():
            acc_correct[task["name"]] += 1
        print(
            f"[{idx:4d}/{len(pairs)}] {task['name']:8s} "
            f"pred={pred:<5} gold={task['gold']}  "
            f"running-acc={acc_correct[task['name']]/acc_total[task['name']]:.3f}"
        )

print("\nFinal accuracies:")
for name in sorted(acc_total):
    print(f"  {name:8s}: {acc_correct[name] / acc_total[name]:.3f}")
