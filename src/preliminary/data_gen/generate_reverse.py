from config.paths import DATA_DIR, RESULTS_DIR, OUTPUTS_DIR, ATTR_SCORES_DIR
#!/usr/bin/env python3
"""
reverse_question_generator_with_progress.py

Rewrite every embedded natural–language logic puzzle in a JSON file so that its
correct answer flips from **True** to **False**, **while tracking live progress
and optionally logging every rewrite**.

New in this version
-------------------
* **Progress bar** – counts prompts ahead of time and shows a tqdm bar.
* **Optional CSV log** – use ``--log-file path/to/log.csv`` to record original
  and rewritten questions plus timestamps.
* **Final summary** – prints totals for prompts rewritten and answers flipped.

Usage
~~~~~
python reverse_question_generator_with_progress.py input.json output.json \
    --model Meta-Llama-3.1-8B-Instruct --rate 1 \
    --flip-all-answers --log-file rewrite_log.csv

Environment
~~~~~~~~~~~
Set the ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` environment variables, *or*
edit the ``client = OpenAI(...)`` line near the bottom.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Tuple

from openai import OpenAI
from tqdm import tqdm
import pdb

# ── 0. Prompt‑splitting helpers ─────────────────────────────────────────

def split_prompt(clean: str) -> Tuple[str, str, str]:
    """Return (preamble, question, suffix) for a full prompt string."""
    suffix_idx = -1
    q_start_idx = clean.rfind("\n", 0, suffix_idx) + 1
    preamble = clean[:q_start_idx]
    question = clean[q_start_idx:suffix_idx].rstrip()
    suffix_tail = clean[suffix_idx:]
    return preamble, question, suffix_tail

# ── 1. LLM wrapper ─────────────────────────────────────────────────────

def make_client() -> OpenAI:
    """Instantiate the OpenAI client using env vars or fallback values."""
    api_key = os.getenv("OPENAI_API_KEY", "dc38c182-a891-43b2-9a4b-87577d072688")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.sambanova.ai/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def rewrite_question(
    client: OpenAI,
    model: str,
    question: str,
    *,
    temperature: float = 0.0,
) -> str:
    """Return a single‑sentence rewrite whose logical answer is *False*.

    Falls back to **manual input** if the model yields an empty response.
    """
    sys_prompt = (
        "You are an expert in formal logic. "
        "Rewrite the user's final interrogative sentence so that, given the same premises, "
        "the original answer will reverse. Keep variable names and overall style similar; "
        "output only the rewritten sentence."
    )
    user_msg = f"Original question:\n{question}\n\nRewritten question:"

    try:
        resp_obj = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
        )
        resp_text = (
            resp_obj.choices[0].message.content.strip() if resp_obj.choices else ""
        )
    except Exception as exc:
        print(f"⚠️  API error ({exc}). Switching to manual input.")
        resp_text = ""

    if not resp_text:
        # --- Interactive fallback ------------------------------------------------
        print("\n⚠️  The model did not return a rewrite for the following question:\n")
        print(question)
        resp_text = input("Please type a rewritten question that flips the answer → ").strip()

    return resp_text



# ── 2. Progress helpers ────────────────────────────────────────────────

def count_prompts(node: Any) -> int:
    """Recursively count how many dicts contain a '<nl>' string."""
    if isinstance(node, dict):
        cnt = 1 if ("<nl>" in node and isinstance(node["<nl>"], str)) else 0
        return cnt + sum(count_prompts(v) for v in node.values())
    if isinstance(node, list):
        return sum(count_prompts(item) for item in node)
    return 0

# ── 3. Recursive JSON traversal ────────────────────────────────────────

def process_prompt_dict(
    dct: Dict[str, Any],
    client: OpenAI,
    model: str,
    *,
    sleep_s: float,
    pb: "tqdm | None",
    csv_writer: "csv.writer | None" = None,
):
    """Rewrite dct["<nl>"] in‑place so that its logical answer flips."""
    original = dct["<nl>"]
    pre, q, suf = split_prompt(original)
    new_q = rewrite_question(client, model, q)
    dct["<nl>"] = f"{pre}{new_q}{suf}"

    if csv_writer:
        csv_writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "original_question": q,
                "rewritten_question": new_q,
            }
        )

    if pb:
        pb.update(1)
    time.sleep(sleep_s)


def recurse(
    node: Any,
    client: OpenAI,
    model: str,
    *,
    sleep_s: float,
    flip_answer: bool,
    pb: "tqdm | None" = None,
    csv_writer: "csv.writer | None" = None,
):
    """Depth‑first in‑place transformation across dicts & lists."""
    if isinstance(node, dict):
        if flip_answer and node.get("answer") in {True, False}:
            node["answer"] = not node["answer"]
        if "<nl>" in node and isinstance(node["<nl>"], str):
            process_prompt_dict(
                node, client, model, sleep_s=sleep_s, pb=pb, csv_writer=csv_writer
            )
        for value in node.values():
            recurse(
                value,
                client,
                model,
                sleep_s=sleep_s,
                flip_answer=flip_answer,
                pb=pb,
                csv_writer=csv_writer,
            )
    elif isinstance(node, list):
        for item in node:
            recurse(
                item,
                client,
                model,
                sleep_s=sleep_s,
                flip_answer=flip_answer,
                pb=pb,
                csv_writer=csv_writer,
            )

# ── 4. Main entry‑point ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reverse‑answer logic questions with progress tracking.",
    )
    parser.add_argument(
        "--input_json",
        default=str(DATA_DIR / "logic/deductive_logic.json"),
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "--output_json",
        default=str(DATA_DIR / "logic/deductive_logic_reverse.json"),
        help="Where to write the transformed JSON",
    )
    parser.add_argument(
        "--model",
        default="Meta-Llama-3.1-8B-Instruct",
        help="Model name for the chat completion API",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls",
    )
    parser.add_argument(
        "--flip-all-answers",
        action="store_true",
        help="Also flip every boolean 'answer' field encountered",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar",
    )
    parser.add_argument(
        "--log-file",
        help="Path to a CSV file recording each rewrite (optional)",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_prompts = count_prompts(data)
    print(f"Found {total_prompts} prompts to rewrite.")

    client = make_client()

    pb = None if args.no_progress else tqdm(total=total_prompts, desc="Rewriting", unit="prompt")

    csv_writer = None
    if args.log_file:
        log_fh = open(args.log_file, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            log_fh, fieldnames=["timestamp", "original_question", "rewritten_question"]
        )
        csv_writer.writeheader()
    else:
        log_fh = None

    recurse(
        data,
        client,
        args.model,
        sleep_s=args.rate,
        flip_answer=args.flip_all_answers,
        pb=pb,
        csv_writer=csv_writer,
    )

    if pb:
        pb.close()
    if log_fh:
        log_fh.close()

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(
        f"Done – transformed file written to {args.output_json} (rewritten {total_prompts} prompts).",
    )


if __name__ == "__main__":
    main()
