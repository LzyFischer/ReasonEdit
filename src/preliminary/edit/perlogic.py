from __future__ import annotations
from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR
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

import argparse
import json
import random
import re
import time
from pathlib import Path
from collections import defaultdict
import copy
import pandas as pd
import pdb

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb  # noqa: F401
from tqdm import tqdm

###############################################################################
# 0. Reproducibility, args & paths
###############################################################################
SEED = 1
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
    default=DATA_DIR / "logic/deductive_logic.json",
    help="Path to source JSON file",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
    help="Learning rate for the optimizer",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen2.5-3B-Instruct",
    help="Name of the base model to use",
)
parser.add_argument(
    "--correct_file",
    type=str,
    default=str(DATA_DIR / "processed/correct_pairs_llama_7b.json"),
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
        prompt = f"{row['text']} (Answer in True, False, or N/A (Neither)). Answer:"
    else:
        prompt = f"{row['text']}. Answer:"
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
            templ = f"{prompt} (Answer in True, False, or N/A (Neither)). Answer:"
        else:
            templ = f"{prompt}\n### The answer is:"
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


###############################################################################
# 3. Build (train, eval) pairs
###############################################################################
POSSIBLE_ANSWERS = ["true", "false", "n/a"]

if args.model_name.startswith("mistralai/Mistral-7B"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_mistral_7b.json")
elif args.model_name.startswith("meta-llama/Llama-2-7b"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_llama_7b.json")
elif args.model_name.startswith("meta-llama/Llama-3.2-3B"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_llama_3b.json")
elif args.model_name.startswith("TinyLlama/TinyLlama-1.1B"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_tinyllama_1b.json")
elif args.model_name.startswith("openai-community/gpt2-large"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_gpt2_large.json")
elif args.model_name.startswith("openai-community/gpt2"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_gpt2.json")
elif args.model_name.startswith("Qwen/Qwen1.5-1.8B-Chat"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_qwen_1_8b_chat.json")
elif args.model_name.startswith("Qwen/Qwen1.5-0.5B"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_qwen_0_5b.json")
elif args.model_name.startswith("Qwen/Qwen1.5-1.8B"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_qwen_1_8b.json")
elif args.model_name.startswith("Qwen/Qwen2-7B-Instruct"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_qwen_2_7b.json")
elif args.model_name.startswith("Qwen/Qwen2.5-7B-Instruct"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_qwen_2_5_7b.json")
elif args.model_name.startswith("google/gemma-7b-it"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_gemma_7b.json")
elif args.model_name.startswith("google/gemma-3-4b-it"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_gemma_3_4b.json")
elif args.model_name.startswith("google/gemma-2-2b-it"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_gemma_2_2b.json")
elif args.model_name.startswith("Qwen/Qwen2.5-1.5B"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_qwen_2_5_1_5b.json")
elif args.model_name.startswith("Qwen/Qwen2.5-3B-Instruct"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_qwen_2_5_3b_it.json")
elif args.model_name.startswith("Qwen/Qwen2.5-3B"):
    args.correct_file = str(DATA_DIR / "processed/correct_pairs_qwen_2_5_3b.json")


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
                    a_random_sample = random.choice(list(topics.values()))['<nl>'].strip()
                    if args.cot:
                        train_text = a_random_sample
                        train_label = gold
                        eval_text = f"{a_random_sample} \n### Answer:" + gold + "\n" + nl
                        eval_label = gold
                    elif args.fine_tune:
                        train_text = a_random_sample
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
                            "logic": glb_prompt,          # <-- CHANGE HERE
                            "train": {"text": train_text, "label": train_label},
                            "eval":  {"prompt": eval_text,  "gold":  eval_label},
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

    # Index: logic → {gold → [pairs]}
    logic_gold_to_pairs = defaultdict(lambda: defaultdict(list))
    for p in pairs_correct:
        logic_gold_to_pairs[p["logic"]][p["eval"]["gold"]].append(p)

    all_logics = sorted(logic_gold_to_pairs.keys())

    for pair in tqdm(pairs, desc="Attaching locality probes"):
        logic0, gold0 = pair["logic"], pair["eval"]["gold"]
        probes = []

        for logic1 in all_logics:
            if logic1 == logic0:
                continue                          # only different logic columns

            # candidates inside logic1 whose gold flips w.r.t. the current pair
            flipped_gold = [g for g in POSSIBLE_ANSWERS if g != gold0]
            candidates = [
                q for g in flipped_gold
                for q in logic_gold_to_pairs[logic1].get(g, [])
            ]

            if not candidates:
                # no answer-flipped example exists for this column
                # print(f"[warn] logic '{logic1[:40]}…' has no flip vs '{logic0[:40]}…'")
                continue

            probes.append(
                {
                    "prompt": random.choice(candidates)["eval"]["prompt"],
                    "gold":   candidates[0]["eval"]["gold"],   # all share same flipped gold
                    "logic":  logic1,
                }
            )

        pair["locality_eval"] = probes   
    return pairs


pairs = harvest_pairs(SRC_JSON, args)
print(f"Loaded {len(pairs)} (train, eval) pairs.")

# --- create short tags ---------------------------------------------------
logic_types      = sorted({p["logic"] for p in pairs})            # full prompts
logic_tags       = [f"logic_{i:03d}" for i in range(len(logic_types))]
tag_of           = {lg: tag for lg, tag in zip(logic_types, logic_tags)}
idx_of           = {lg: i   for i, lg in enumerate(logic_types)}

mat_total   = [[0]*len(logic_types) for _ in logic_types]
mat_correct = [[0]*len(logic_types) for _ in logic_types]


###############################################################################
# 4. Train‐then‐eval loop
###############################################################################
print("\nStarting sequential training…")
start = time.time()

# running counters (optional)
hits_gen = 0
hits_loc = 0
gen_total = 0
loc_total = 0

# LORA_PAT = re.compile(r"lora\.")
# orig_lora_state = {
#     name: p.detach().clone()
#     for name, p in model.named_parameters()
#     if LORA_PAT.search(name)
# }
orig_lora_state = {
    name: p.detach().clone()
    for name, p in model.named_parameters()
    if p.requires_grad          # ← catches lora_A, lora_B, lora_embedding, etc.
}
print(f"Cached {len(orig_lora_state)} LoRA tensors.")

for step, pair in enumerate(pairs, 1):
    row = idx_of[pair["logic"]]           # training-logic index

    # ── (a) reset only the LoRA params ───────────────────────────────────
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in orig_lora_state:
                p.copy_(orig_lora_state[name])
    # optimizer.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # ── (b) build batch & (c) few-step inner FT ──────────────────────────
    batch = {k: v.unsqueeze(0).to(device) for k, v in encode(pair["train"]).items()}

    loss = torch.tensor(0.0, device=device)
    model.train()
    for _ in range(10 if args.fine_tune else 0):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # ── (d) generality evaluation (diagonal) ─────────────────────────────
    model.eval()
    gen_pred = generate_answer(pair["eval"]["prompt"], model, max_new_tokens=16)
    gen_correct = pair["eval"]["gold"] in gen_pred

    mat_total[row][row] += 1
    gen_total += 1
    if gen_correct:
        mat_correct[row][row] += 1
        hits_gen += 1

    # ── (e) locality evaluation (one probe per *other* logic) ────────────
    for loc in pair.get("locality_eval", []):
        col = idx_of[loc["logic"]]        # evaluation-logic index

        loc_pred = generate_answer(loc["prompt"], model, max_new_tokens=16)
        loc_correct = loc["gold"] in loc_pred

        mat_total[row][col] += 1
        loc_total += 1
        if loc_correct:
            mat_correct[row][col] += 1
            hits_loc += 1

    # ── (f) optional progress printout ───────────────────────────────────
    if step % 10 == 0 or step == len(pairs):
        print(
            f"[{step:4d}/{len(pairs)}]  "
            f"gen_acc={hits_gen/gen_total:.3f}  "
            f"loc_acc={hits_loc/loc_total:.3f}  "
            f"loss={loss.item():.4f}",
            flush=True,
        )

# 5-a. accuracy matrix (float), plus raw counts
acc = [
    [
        (mat_correct[r][c] / mat_total[r][c]) if mat_total[r][c] else 0.0
        for c in range(len(logic_types))
    ]
    for r in range(len(logic_types))
]

df_acc     = pd.DataFrame(acc,       index=logic_tags, columns=logic_tags)
df_total   = pd.DataFrame(mat_total, index=logic_tags, columns=logic_tags)
df_correct = pd.DataFrame(mat_correct,index=logic_tags, columns=logic_tags)

OUTPUTS_DIR / "perlogic".mkdir(parents=True, exist_ok=True)
df_acc    .to_csv(str(OUTPUTS_DIR / "perlogic/accuracy.csv"),  float_format="%.4f")
df_total  .to_csv(str(OUTPUTS_DIR / "perlogic/n_total.csv"))
df_correct.to_csv(str(OUTPUTS_DIR / "perlogic/n_correct.csv"))

# save the tag ↔︎ prompt lookup table
pd.DataFrame({"tag": logic_tags, "prompt": logic_types}) \
  .to_csv(str(OUTPUTS_DIR / "perlogic/logic_tags.csv"), index=False)

print("\nSaved to output/perlogic/:")
print("  • accuracy.csv   (numeric accuracy matrix)")
print("  • n_total.csv    (#evals per cell)")
print("  • n_correct.csv  (#correct per cell)")
print("  • logic_tags.csv (tag → original prompt)")