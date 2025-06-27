#!/usr/bin/env python
"""
Sequentially fine-tune a LoRA adapter on one (train, eval) pair at a time and
immediately evaluate on the corresponding full-question prompt. After each
pair, the script prints the running accuracy over all processed pairs and
finally saves per-logic accuracy matrices.

Re-organised for clarity and maintainability:
• Clear top-level constants and configuration helpers
• All IO / CLI parsing grouped in `get_args()`
• Stateless utility helpers collected together
• Model and tokenizer initialisation encapsulated in `get_model()`
• The sequential training loop moved into a `Trainer` class
• Metrics handling extracted to dedicated helpers
• `main()` entry-point with guard for import safety
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

###############################################################################
# 0. Global constants & reproducibility                                       #
###############################################################################
SEED = 1
MAX_LEN = 256
POSSIBLE_ANSWERS = ["true", "false", "n/a"]

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

###############################################################################
# 1. Configuration & CLI                                                      #
###############################################################################

def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes"}:
        return True
    if v.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential LoRA fine-tuning on logic pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # training behaviour
    parser.add_argument("--fine_tune",  type=str2bool, default=True)
    parser.add_argument("--lr",         type=float,    default=1.5e-4,
                        help="Learning rate for the LoRA parameters")

    # dataset paths
    parser.add_argument("--src_json",   type=Path,     default=Path("data/logic/deductive_logic.json"))
    parser.add_argument("--correct_file", type=str,
                        default="data/processed/correct_pairs_llama_7b.json")

    # sampling-scheme
    parser.add_argument("--gen_k", type=int, default=1,
                        help="# generative prompts from SAME logic")
    parser.add_argument("--loc_k", type=int, default=1,
                        help="# locality probes from EACH OTHER logic")

    # model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")

    return parser.parse_args()

###############################################################################
# 2. Tokenizer, encoding & generation helpers                                #
###############################################################################

def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode_example(row: dict, tokenizer, src_path: Path):
    """Convert a (premise, answer) example into supervised tokens."""
    if "deductive" in src_path.name:
        prompt = f"{row['text']} (Answer in True, False, or N/A (Neither)). Answer:"
    else:
        prompt = f"{row['text']}. Answer:"
    answer = f"{row['label']}"
    full = prompt + " " + answer + tokenizer.eos_token

    ids = tokenizer(full, max_length=MAX_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    input_ids  = ids["input_ids"].squeeze(0)
    attn_mask  = ids["attention_mask"].squeeze(0)
    labels     = input_ids.clone()

    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0).numel()
    labels[:prompt_len] = -100
    labels[attn_mask == 0] = -100

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def generate_answer(prompt: str, model, tokenizer, src_path: Path, max_new: int = 5) -> str:
    """Greedy decode a short answer and normalise to {true,false,n/a}."""
    model.eval()
    with torch.no_grad():
        if "deductive" in src_path.name:
            templ = f"{prompt} (Answer in True, False, or N/A (Neither)). Answer:"
        else:
            templ = f"{prompt}\n### The answer is:"
        ids = tokenizer(templ, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)

    # crude normalisation heuristics
    answer = text.split(" Answer:")[-1].strip().split()[0].lower()
    if not any(k in answer for k in POSSIBLE_ANSWERS):
        lowered = text.split(" Answer:")[-1].lower()
        if any(x in lowered for x in {"yes", "true"}):
            answer = "true"
        elif any(x in lowered for x in {"not", "no", "false"}):
            answer = "false"
        elif "n/a" in lowered:
            answer = "n/a"
        else:
            answer = "true"  # default fall-back
    return answer

###############################################################################
# 3. Data preparation                                                         #
###############################################################################
CUE = re.compile(
    r"\b(?:then|which implies|this (?:would )?implies?|would suggest that|"
    r"implies?|suggests? that)\b", re.I,
)


def split_premise(nl: str):
    """Return (antecedent, consequent) if the sentence has a clear split."""
    first, *rest = nl.split("\n", 2)
    premise = (rest[0] if rest else first).strip()
    if m := CUE.search(premise):
        ante, cons = CUE.split(premise, 1)
        return ante.strip(", ; "), m.group(0) + cons.strip()
    sents = re.split(r"(?<=\.|!|\?)\s+", premise)
    return (" ".join(sents[:-1]).strip(), sents[-1].strip()) if len(sents) > 1 else (None, None)


def harvest_pairs(path: Path, gen_k: int, loc_k: int):
    """Load dataset and attach generativity / locality probes."""
    examples = []
    pool_by_logic = defaultdict(list)

    # –– load dataset ––
    for rec in json.load(path.open()):
        logic = rec["question"][0]["<nl>"].strip()
        gold  = str(rec["answer"]).lower()
        for dom, topics in rec.items():
            if dom in {"question", "answer"}:
                continue
            for payload in topics.values():
                nl_full  = payload["<nl>"].strip()
                premise  = nl_full.split("\nGiven")[0].strip()
                question = nl_full.split("\n")[-1].strip()
                ex = {
                    "logic": logic,
                    "train": {"text": premise, "label": gold},
                    "eval":  {"prompt": question, "gold": gold},
                }
                examples.append(ex)
                pool_by_logic[logic].append(ex)

    rng = random.Random(SEED)
    for ex in examples:
        logic = ex["logic"]
        # generativity – same-logic pool (excluding itself)
        same_pool = [e for e in pool_by_logic[logic] if e is not ex]
        ex["gen_eval"] = random.sample(same_pool, k=min(gen_k, len(same_pool)))
        # locality – LOC_K from each other logic
        loc_probes = []
        for other_logic, buf in pool_by_logic.items():
            if other_logic == logic:
                continue
            loc_probes.extend(random.sample(buf, k=min(loc_k, len(buf))))
        ex["loc_eval"] = loc_probes
    return examples

###############################################################################
# 4. Model & LoRA initialisation                                             #
###############################################################################

def get_model(model_name: str, tokenizer, device: torch.device):
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    # target-module heuristic
    target_modules = [
        "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
    ] if model_name.startswith("google/gemma-3-4b-it") else None

    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=target_modules)
    model = get_peft_model(base, lora_cfg).to(device)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model

###############################################################################
# 5. Trainer                                                                 #
###############################################################################
class Trainer:
    def __init__(self, args, tokenizer, model, pairs):
        self.args = args
        self.tok = tokenizer
        self.model = model
        self.pairs = pairs
        self.device = model.device
        self._init_metrics()
        # snapshot initial LoRA weights
        self.orig_lora_state = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---------------------------------------------------------------------
    # metrics helpers
    # ---------------------------------------------------------------------
    def _init_metrics(self):
        logic_types = sorted({p["logic"] for p in self.pairs})
        self.logic_types = logic_types
        self.tag_of = {lg: f"logic_{i:03d}" for i, lg in enumerate(logic_types)}
        self.idx_of = {lg: i for i, lg in enumerate(logic_types)}
        n = len(logic_types)
        self.mat_total   = [[0]*n for _ in range(n)]
        self.mat_correct = [[0]*n for _ in range(n)]
        self.mat_dgen    = [[0]*n for _ in range(n)]
        self.mat_dloc    = [[0]*n for _ in range(n)]
        # running
        self.hits_gen = self.hits_loc = self.gen_total = self.loc_total = 0

    # ---------------------------------------------------------------------
    # train-evaluate loop
    # ---------------------------------------------------------------------
    def run(self):
        print("\nStarting sequential training…")
        start = time.time()
        for step, pair in enumerate(self.pairs, 1):
            continue
            self._reset_lora()
            self._process_pair(pair)
            if step % 2 == 0 or step == len(self.pairs):
                print(self._progress_str(step))
        print(f"Completed in {(time.time()-start)/60:.1f} min")

        lr_slug = str(self.args.lr).replace('.', 'p').replace('-', 'm')
        out_dir = Path(f"output/perlogic/{lr_slug}")
        self._save_results(out_dir)

    # ------------------------------------------------------------------
    def _reset_lora(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.orig_lora_state:
                    p.copy_(self.orig_lora_state[n])

    # ------------------------------------------------------------------
    def _process_pair(self, pair: dict):
        row = self.idx_of[pair["logic"]]
        # -- pre-train predictions
        pre_gen = generate_answer(pair["eval"]["prompt"], self.model, self.tok, self.args.src_json)
        pre_hits_gen = [
            p["gold"] in generate_answer(p["prompt"], self.model, self.tok, self.args.src_json)
            for p in pair["gen_eval"]
        ]
        pre_hits_loc = [
            loc["gold"] in generate_answer(loc["prompt"], self.model, self.tok, self.args.src_json)
            for loc in pair["loc_eval"]
        ]
        # fine-tune on single example
        if self.args.fine_tune:
            batch = {k: v.unsqueeze(0).to(self.device) for k, v in
                     encode_example(pair["train"], self.tok, self.args.src_json).items()}
            for _ in range(10):
                loss = self.model(**batch).loss
                loss.backward(); self.optimizer.step(); self.optimizer.zero_grad()
        else:
            loss = torch.tensor(0.0)

        # -- post-train predictions
        post_hits_gen = [
            p["gold"] in generate_answer(p["prompt"], self.model, self.tok, self.args.src_json)
            for p in pair["gen_eval"]
        ]
        post_hits_loc = [
            loc["gold"] in generate_answer(loc["prompt"], self.model, self.tok, self.args.src_json)
            for loc in pair["loc_eval"]
        ]
        # metrics update
        self._update_matrices(row, pre_hits_gen, post_hits_gen,
                              pair["loc_eval"], pre_hits_loc, post_hits_loc)
        self.last_loss = loss.item()

    # ------------------------------------------------------------------
    def _update_matrices(self, row, pre_g, post_g, locs, pre_h, post_h):
        for a, b in zip(pre_g, post_g):
            self._record(row, row, a, b, gen=True)
        for loc, a, b in zip(locs, pre_h, post_h):
            col = self.idx_of[loc["logic"]]
            self._record(row, col, a, b, gen=False)

    def _record(self, r, c, pre, post, *, gen: bool):
        self.mat_total[r][c]   += 1
        self.mat_correct[r][c] += int(post)
        if gen and r == c:
            self.mat_dgen[r][c] += abs(int(post) - int(pre))
            self.gen_total += 1
            self.hits_gen  += int(post)
        elif not gen:
            self.mat_dloc[r][c] += abs(int(post) - int(pre))
            self.loc_total += 1
            self.hits_loc  += int(post)

    # ------------------------------------------------------------------
    def _progress_str(self, step):
        g_acc = self.hits_gen / self.gen_total if self.gen_total else 0
        l_acc = self.hits_loc / self.loc_total if self.loc_total else 0
        return (f"[{step:4d}/{len(self.pairs)}]  "
                f"gen_acc={g_acc:.3f}  loc_acc={l_acc:.3f}  loss={self.last_loss:.4f}")

    # ------------------------------------------------------------------
    def _save_results(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        # accuracy matrix
        acc = [[(self.mat_correct[r][c] / self.mat_total[r][c])
                if self.mat_total[r][c] else 0.0
                for c in range(len(self.logic_types))]
               for r in range(len(self.logic_types))]

        tags = [self.tag_of[lg] for lg in self.logic_types]
        delta_all = [[self.mat_dgen[r][c] if r == c else self.mat_dloc[r][c]
                       for c in range(len(self.logic_types))]
                      for r in range(len(self.logic_types))]
        delta_all = np.array(delta_all) / np.array(self.mat_total)

        pd.DataFrame(acc, index=tags, columns=tags).to_csv(out_dir/"accuracy.csv", float_format="%.4f")
        pd.DataFrame(self.mat_total,   index=tags, columns=tags).to_csv(out_dir/"n_total.csv")
        pd.DataFrame(self.mat_correct, index=tags, columns=tags).to_csv(out_dir/"n_correct.csv")        
        pd.DataFrame(delta_all, index=tags, columns=tags).to_csv(out_dir/"delta_all.csv")

        # tag lookup
        pd.DataFrame({"tag": tags, "prompt": self.logic_types}).to_csv(out_dir/"logic_tags.csv", index=False)
        print("\nSaved metrics to", out_dir)

###############################################################################
# 6. Main                                                                    #
###############################################################################

def main():
    set_seed()
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer(args.model_name)
    model = get_model(args.model_name, tokenizer, device)
    print("Loaded model with trainable LoRA parameters:")
    model.print_trainable_parameters()

    pairs = harvest_pairs(args.src_json, args.gen_k, args.loc_k)
    print(f"Loaded {len(pairs)} (train, eval) pairs.")

    trainer = Trainer(args, tokenizer, model, pairs)
    trainer.run()


if __name__ == "__main__":
    main()
