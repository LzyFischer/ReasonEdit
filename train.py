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
    default=Path("data/deductive_logic.json"),
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
    default="mistralai/Mistral-7B-Instruct-v0.3",
    help="Name of the base model to use",
)
parser.add_argument(
    "--correct_file",
    type=str,
    default="data/processed/correct_pairs_llama_7b.json",
    help="Path to save/load correct pairs",
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
    """Convert a (premise, answer) example into supervised tokens."""
    if "deductive" in SRC_JSON.name:
        prompt = f"{row['text']} (True, False, or N/A (undetermined)).\n### Answer:"
    else:
        prompt = f"{row['text']}\n### Answer:"
    answer = f"{row['label']}"
    full = prompt + " " + answer + tokenizer.eos_token

    ids = tokenizer(
        full,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = ids["input_ids"].squeeze(0)
    attn_mask = ids["attention_mask"].squeeze(0)
    labels = input_ids.clone()

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0).numel()
    labels[:prompt_len] = -100
    labels[attn_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }


def generate_answer(prompt: str, model, max_new_tokens: int = 50) -> str:
    model.eval()
    with torch.no_grad():
        if "deductive" in SRC_JSON.name:
            templ = f"{prompt} (True, False, or N/A (undetermined)).\n### Answer:"
        else:
            templ = f"{prompt}\n### Answer:"
        ids = tokenizer(templ, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        try:
            answer = text.split("### Answer:")[-1].strip().split()[0].lower()
            if "true" not in answer and "false" not in answer and "n/a" not in answer:
                # extract the "not" as false, else as true after the "### Answer:"
                answers = text.split("### Answer:")[-1]
                if "not" in answers:
                    answer = "false"
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

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_cfg).to(device)
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


###############################################################################
# 3. Build (train, eval) pairs
###############################################################################
if args.model_name.startswith("mistralai/Mistral-7B"):
    args.correct_file = "data/processed/correct_pairs_mistral_7b.json"
elif args.model_name.startswith("meta-llama/Llama-2-7b"):
    args.correct_file = "data/processed/correct_pairs_llama_7b.json"
elif args.model_name.startswith("meta-llama/Llama-3.2-3B"):
    args.correct_file = "data/processed/correct_pairs_llama_3b.json"
elif args.model_name.startswith("TinyLlama/TinyLlama-1.1B"):
    args.correct_file = "data/processed/correct_pairs_tinyllama_1b.json"
elif args.model_name.startswith("openai-community/gpt2"):
    args.correct_file = "data/processed/correct_pairs_gpt2.json"


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


def harvest_pairs(path: Path, args):
    pairs: list[dict] = []
    for rec in json.load(path.open()):
        glb_prompt = rec["question"][0]["<nl>"].strip()  # the reasoning prompt
        gold = str(rec["answer"]).lower()
        for dom, topics in rec.items():
            if dom in {"question", "answer"}:
                continue
            for payload in topics.values():
                nl = payload["<nl>"]
                premise_part = nl.split("\nGiven")[0]
                reason_part = nl.rstrip().split("\n")[-1]
                ant, cons = split_premise(premise_part)
                nl = nl.rstrip("\n")

                if args.quest:
                    train_text = glb_prompt
                    train_label = gold
                    eval_text = reason_part
                    eval_label = gold
                elif args.knowledge:
                    if args.cot:
                        train_text = glb_prompt
                        train_label = gold
                        eval_text = nl
                        eval_label = gold
                    elif args.fine_tune:
                        train_text = ant
                        train_label = cons
                        eval_text = reason_part
                        eval_label = gold
                elif args.reason:
                    if args.cot:
                        train_text = glb_prompt
                        train_label = gold
                        eval_text = f"{glb_prompt} \n### Answer:" + gold + "\n" + nl
                        eval_label = gold
                    elif args.fine_tune:
                        train_text = glb_prompt
                        train_label = gold
                        eval_text = nl
                        eval_label = gold
                elif args.gt:
                    if args.cot:
                        train_text = glb_prompt
                        train_label = gold
                        eval_text = f"{nl} \n### Answer:" + gold + "\n" + nl
                        eval_label = gold
                    elif args.fine_tune:
                        train_text = nl
                        train_label = gold
                        eval_text = nl
                        eval_label = gold

                if ant and cons:
                    pairs.append(
                        {
                            "train": {"text": train_text, "label": train_label},
                            "eval": {"prompt": eval_text, "gold": eval_label},
                        }
                    )

    # first find all pairs that are correct, save it, if there is already a file, then load it
    if Path(args.correct_file).exists():
        with open(args.correct_file, "r") as f:
            pairs_correct = json.load(f)
    else:
        pairs_correct = []
        for pair in tqdm(pairs, desc="Finding correct pairs"):
            pred = generate_answer(pair["eval"]["prompt"], model)
            correct = pred.startswith(pair["eval"]["gold"])
            # correct = True  # Assume all pairs are correct for now
            if correct:
                pairs_correct.append(pair)
        with open(args.correct_file, "w") as f:
            json.dump(pairs_correct, f, indent=2)
        print(f"Found {len(pairs_correct)} correct pairs.")

    # build edit and locality
    random.shuffle(pairs)
    for idx, pair in tqdm(enumerate(pairs), total=len(pairs)):
        # randomly select a pair that is correct and has different label with the current pair
        try:
            other_pair = random.choice(
                [p for p in pairs_correct if p["eval"]["gold"] != pair["eval"]["gold"]]
            )
        except IndexError:
            other_pair = random.choice(pairs_correct)
        pair["locality_eval"] = {
            "prompt": other_pair["eval"]["prompt"],
            "gold": other_pair["eval"]["gold"],
        }
    return pairs


pairs = harvest_pairs(SRC_JSON, args)
print(f"Loaded {len(pairs)} (train, eval) pairs.")


###############################################################################
# 4. Train‐then‐eval loop
###############################################################################
print("\nStarting sequential training…")
start = time.time()
hits = 0
hits_local = 0

LORA_PAT = re.compile(r"lora_(A|B)\.")
orig_lora_state = {
    name: p.detach().clone()
    for name, p in model.named_parameters()
    if LORA_PAT.search(name)
}
print(f"Cached {len(orig_lora_state)} LoRA tensors.")

for idx, pair in enumerate(pairs, 1):
    # (a) reset only the LoRA parameters
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in orig_lora_state:
                p.copy_(orig_lora_state[name])
    optimizer.zero_grad(set_to_none=True)

    # (b) prepare batch
    batch = encode(pair["train"])
    batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}

    # (c) few‐step inner fine‐tuning
    model.train()
    if args.fine_tune:
        epochs = 10
    else:
        epochs = 0
    for _ in range(epochs):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if not args.fine_tune:
        loss = torch.tensor(0.0, device=device)

    # evaluation
    pred = generate_answer(pair["eval"]["prompt"], model)
    correct = pred.startswith(pair["eval"]["gold"])
    hits += int(correct)
    print(
        f"[{idx:4d}/{len(pairs)}] loss={loss.item():.4f}  pred={pred:<5}  "
        f"gold={pair['eval']['gold']}  running‐acc={hits/idx:.3f}",
        flush=True,
    )

    # locality evaluation
    local_pred = generate_answer(
        pair["locality_eval"]["prompt"], model, max_new_tokens=16
    )
    local_correct = local_pred.startswith(pair["locality_eval"]["gold"])
    hits_local += int(local_correct)
    print(
        f"Locality eval: pred={local_pred:<16}  "
        f"gold={pair['locality_eval']['gold']}  "
        f"correct={local_correct}  running‐acc={hits_local/idx:.3f}",
        flush=True,
    )

###############################################################################
# 5. Save adapter & tokenizer
###############################################################################
SAVE_DIR = "output/lora_logic_pairwise"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(
    f"\nFinished in {time.time() - start:.1f}s.  Final accuracy: {hits/len(pairs):.3f}.\n"
    f"Adapter saved to {SAVE_DIR}."
)
