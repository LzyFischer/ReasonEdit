import json, re
from pathlib import Path
from openai import OpenAI
from transformers import AutoTokenizer
from typing import Tuple, List, Dict
import pdb

###############################################################################
# 1.  Config
###############################################################################
INPUT_FILE  = Path("logic/deductive_logic.json")
OUTPUT_FILE = Path("augmented_dataset.json")

MODEL_NAME  = "Meta-Llama-3.1-8B-Instruct"
SEED        = 42
client = OpenAI(
    api_key="dc38c182-a891-43b2-9a4b-87577d072688",
    base_url="https://api.sambanova.ai/v1",
)

TOKENIZER_PATH = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer      = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

###############################################################################
# 2.  Helpers
###############################################################################
ANSWER_MARK      = "### The answer is:"
SUFFIX_NO_NL     = " (True, False, or N/A)."         # ← WITHOUT the trailing newline
SUFFIX_WITH_MARK = SUFFIX_NO_NL + "\n" + ANSWER_MARK

def token_len(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def truth_label(v) -> str:
    return {True: " True", False: " False", "N/A": " N/A"}[v]

def flip_label(lab: str) -> str:
    return {" True": " False", " False": " True", " N/A": " N/A"}[lab]

def make_clean(nl: str) -> str:
    return nl.strip() + SUFFIX_WITH_MARK

# ────────────────────────────────────────────────────────────────────────────
# Split a clean prompt into:  preamble, question, suffix.
#   clean =  <premise…>\n<question>? (True, False, or N/A).\n### The answer is:
# ────────────────────────────────────────────────────────────────────────────
def split_prompt(clean: str) -> Tuple[str, str, str]:
    suffix_idx   = clean.rfind(SUFFIX_NO_NL)          # first char of suffix
    if suffix_idx == -1:
        raise ValueError("suffix not found")
    q_start_idx  = clean.rfind('\n', 0, suffix_idx) + 1
    preamble     = clean[:q_start_idx]                # includes trailing '\n'
    question     = clean[q_start_idx:suffix_idx].rstrip()
    suffix_tail  = clean[suffix_idx:]                 # suffix + ANSWER_MARK
    return preamble, question, suffix_tail

# ────────────────────────────────────────────────────────────────────────────
# Only ask the LLM to rewrite the question line.
# ────────────────────────────────────────────────────────────────────────────
def corrupt_question(question: str, correct: str) -> str:
    wrong = flip_label(correct)
    sys_msg = (
        "You will receive one yes/no question (the entire line). "
        f"The correct answer is currently {correct.strip()}; "
        f"rewrite MINIMALLY so the correct answer becomes {wrong.strip()}. "
        "Constraints:\n"
        "  • Change as few contiguous tokens as possible (ideally one antonym).\n"
        "  • Keep the SAME NUMBER of whitespace-separated tokens.\n"
        "Return ONLY the rewritten question line and nothing else."
    )
    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": question},
        ],
        temperature=0.0,
        seed=SEED,
    )
    question = rsp.choices[0].message.content.strip()
    THINK_TOKEN = '\n</think>\n\n'
    if THINK_TOKEN in question:
        # If the LLM added a think block, remove it
        question = question.split(THINK_TOKEN)[-1].strip()
    return question


###############################################################################
# 3.  Main loop
###############################################################################
mismatches: List[Dict] = []          # collect problematic items

def process_entry(entry: dict) -> dict:
    new_item = {"word_idxs": {"start": 0, "end": 0}, "prompts": []}
    gt = truth_label(entry.get("answer"))
    if gt == " N/A":
        return new_item

    for group, topics in entry.items():
        if group in {"question", "answer"}:
            continue
        for topic, body in topics.items():
            clean  = make_clean(body["<nl>"])
            pre, q_line, suf = split_prompt(clean)

            # --- 1) 先让 LLM 试一次 ----------------------------------------
            corrupt_q = corrupt_question(q_line, gt)
            corrupt   = pre + corrupt_q + suf

            # --- 2) 如有不匹配 → 交互式修正 -------------------------------
            target_len = token_len(clean)
            while token_len(corrupt) != target_len:
                print("\n❌  Token mismatch:")
                print(f"   Clean   ({target_len}): {q_line}")
                print(f"   Corrupt ({token_len(corrupt)}): {corrupt_q}")
                print("▲ 请手动输入新的『问题行』(直接回车放弃、使用 clean)：")
                user_q = input(">>> ").strip()
                if not user_q:                 # 用户想跳过
                    corrupt   = clean
                    break
                corrupt_q = user_q
                corrupt   = pre + corrupt_q + suf   # 重新组装

            mismatch_flag = (token_len(clean) != token_len(corrupt))
            if mismatch_flag:
                mismatches.append({
                    "group": group, "topic": topic,
                    "clean_q": q_line, "corrupt_q": corrupt_q,
                    "error": "manual skip",
                })

            new_item["prompts"].append({
                "clean":          clean,
                "corrupt":        corrupt,
                "answers":        [gt],
                "wrong_answers":  [flip_label(gt)],
                "token_mismatch": mismatch_flag,
            })

    return new_item

###############################################################################
# 4.  Run
###############################################################################
def main() -> None:
    raw = json.loads(INPUT_FILE.read_text())
    if isinstance(raw, dict):
        raw = [raw]

    new_ds = [process_entry(ent) for ent in raw]
    OUTPUT_FILE.write_text(json.dumps(new_ds, indent=2, ensure_ascii=False))
    print(f"✅  Saved {len(new_ds)} entries to {OUTPUT_FILE}")

    if mismatches:
        print(f"\n⚠️  {len(mismatches)} prompts need manual fixes (see mismatch_log.json)")
        Path("mismatch_log.json").write_text(json.dumps(mismatches, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()