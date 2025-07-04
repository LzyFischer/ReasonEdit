from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR
#!/usr/bin/env python
from __future__ import annotations
import json, re, argparse, random
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer
import pdb

from itertools import count
from tqdm import tqdm               # progress bar
import logging
logging.basicConfig(
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ──────────── CLI ────────────
cli = argparse.ArgumentParser(description="First-premise corruptions")
cli.add_argument("--in_file",  type=Path, default=DATA_DIR / "logic/deductive_logic.json")
cli.add_argument("--out_file", type=Path, default=DATA_DIR / "corrupt/augmented_dataset.json")
cli.add_argument("--tok_path", type=str,
                 default="Qwen/Qwen2.5-3B-Instruct",
                 help="Tokenizer path (only for token-count bookkeeping)")
args = cli.parse_args()
tok  = AutoTokenizer.from_pretrained(args.tok_path, trust_remote_code=True)

ANSWER_MARK      = "Answer:"
SUFFIX_NO_NL     = " (Answer in True, False, or N/A (Neither))."
SUFFIX_WITH_MARK = SUFFIX_NO_NL + " " + ANSWER_MARK

# -------------------------------------------------------------------------
# add once, near the top (right after you load the main tokenizer `tok`)

def tlen1(text: str) -> int: 
    """Token length under the main tokenizer `tok`."""
    return len(tok.encode(text, add_special_tokens=False))


def equalise_len(q_text: str, target_line: str) -> str:
    """
    Edit q_text until tlen1(q_text) == tlen1(target_line).
      • If too long   → delete stop-words first, then tail tokens.
      • If too short  → append '?'  one by one.
    """
    stopwords = {
        "the", "a", "an", "that", "this", "these", "those",
        "did", "does", "do", "is", "are", "was", "were", "has",
    }
    words = q_text.split()
    tq, tt = tlen1(q_text), tlen1(target_line)

    # -------- truncate --------
    if tq > tt:
        i = 0
        while tq > tt and i < len(words):
            w = words[i].lower().strip(".,;:'\"!?")
            if w in stopwords:
                del words[i]
                tq = tlen1(" ".join(words))
            else:
                i += 1
        while tq > tt and words:
            words.pop()
            tq = tlen1(" ".join(words))

    # -------- pad --------
    while tq < tt:
        words[-1] += "?"
        tq = tlen1(" ".join(words))

    return " ".join(words)

def truth_label(v) -> str:
    return {True: " True", False: " False", "N/A": " N"}[v]

def flipped(lab: str) -> str:
    return {" True": " False", " False": " True", " N": " True"}[lab]

def token_len(txt: str) -> int:
    return len(tok.encode(txt, add_special_tokens=False))

# ────────── premise helpers ──────────
NEG_PAT = re.compile(r"\bnot\b|\bn't\b", re.I)
AUX_PAT = re.compile(r"\b(is|are|was|were|has|have|had|can|could|shall|should|"
                     r"will|would|may|might|must|do|does|did)\b", re.I)

def negate(sent: str) -> str:
    if NEG_PAT.search(sent):
        return NEG_PAT.sub("", sent, 1).replace("  ", " ").strip()
    if (aux := AUX_PAT.search(sent)):
        return sent[:aux.end()] + " not" + sent[aux.end():]
    subj = re.match(r"(\b\w+\b(?:\s+\b\w+\b)?)\s+", sent)
    return (sent[:subj.end()] + "didn’t " + sent[subj.end():]) if subj else "not " + sent

def first_premise(payload: dict, preamble: str) -> str:
    if (prem := payload.get("<aaa>")):          # tagged first premise
        return prem.strip().rstrip(".!?")
    # fallback: first non-blank line in the preamble
    for line in preamble.split("\n"):
        if line.strip():
            return line.strip().rstrip(".!?")
    raise ValueError("Could not locate first premise")

# ────────── build new prompts ──────────
mismatches: List[Dict] = []

def make_clean(nl: str) -> str:
    return nl.strip() + SUFFIX_WITH_MARK

def split_prompt(clean: str):
    suf_idx = clean.rfind(SUFFIX_NO_NL)
    q_start = clean.rfind('\n', 0, suf_idx) + 1
    return clean[:q_start], clean[q_start:suf_idx].rstrip(), clean[suf_idx:]

def process_entry(rec: dict, pbar) -> dict:
    gt = truth_label(rec.get("answer", "N/A"))
    out = {"word_idxs": {"start": 0, "end": 0}, "prompts": []}

    for domain in filter(lambda k: k not in {"question", "answer"}, rec):
        for topic, payload in rec[domain].items():
            clean = make_clean(payload["<nl>"])
            pre, q_orig, _ = split_prompt(clean)
            premise = first_premise(payload, pre)

            raw_q = negate(premise) + "?" if gt == " True" else premise + "?"
            q_text = equalise_len(raw_q, q_orig)
            corrupt = pre + q_text + SUFFIX_WITH_MARK

            mm = token_len(clean) != token_len(corrupt)
            if mm:
                mismatches.append(
                    {"domain": domain, "topic": topic,
                     "orig_len": token_len(clean), "new_len": token_len(corrupt)}
                )

            out["prompts"].append({
                "clean":          clean,
                "corrupt":        corrupt,
                "answers":        [gt],
                "wrong_answers":  [flipped(gt)],
                "token_mismatch": mm,
            })
            pbar.update()              # advance progress bar
            if (cnt := next(counter)) % 250 == 0:
                logging.info(f"generated {cnt:,} corrupted prompts so far")
    return out

# ────────── main ──────────
def main() -> None:
    raw = json.load(args.in_file.open())
    if isinstance(raw, dict):
        raw = [raw]

    global counter
    counter = count(1)

    # total inner-loops = sum of topic counts over all records
    total = sum(
        sum(len(rec[d]) for d in rec if d not in {"question", "answer"})
        for rec in raw
    )
    with tqdm(total=total, desc="corrupting") as pbar:
        dataset = [process_entry(r, pbar) for r in raw]

    args.out_file.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"✅  Saved {len(dataset)} items → {args.out_file}")
    if mismatches:
        Path("mismatch_log.json").write_text(json.dumps(mismatches, indent=2, ensure_ascii=False))
        print(f"⚠️  {len(mismatches)} token-length mismatches logged.")

if __name__ == "__main__":
    random.seed(42)
    main()