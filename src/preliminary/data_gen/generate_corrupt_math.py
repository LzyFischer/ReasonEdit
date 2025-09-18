#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 {template_id: [ {problem, result}, ... ]} 转为目标列表格式，并生成 corrupt 版本：
- 句子用逗号/句号分割，取第一句与最后一句
- 用第一句的 token 片段（长度=最后一句的 token 长度）替换掉最后一句
- 若第一句长度不足，用'?'按 tokenizer 粒度补齐；若过长则截断
- token 长度以 Qwen/Qwen-3B-Instruct tokenizer 为准；无法加载时回退字符级
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import math
import random

SPLIT_RE = re.compile(r'([,.])')  # 保留分隔符

def split_preserve(text: str) -> List[str]:
    # e.g. "A, B. C" -> ["A", ",", " B", ".", " C"]
    return SPLIT_RE.split(text)

def first_text_idx(parts: List[str]) -> Optional[int]:
    for i, p in enumerate(parts):
        if p not in {',', '.'} and p.strip() != "":
            return i
    return None

def last_text_idx(parts: List[str]) -> Optional[int]:
    for i in range(len(parts)-1, -1, -1):
        if parts[i] not in {',', '.'} and parts[i].strip() != "":
            return i
    return None

# ---------------- tokenizer wrapper ----------------
class TokWrap:
    def __init__(self, model_name: str):
        self.ok = False
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.ok = True
        except Exception as e:
            print(f"[warn] tokenizer load failed ({e}); fallback to char mode", file=sys.stderr)
            self.tok = None
            self.ok = False

    def encode_ids(self, text: str) -> List[int]:
        if self.ok:
            return self.tok.encode(text, add_special_tokens=False)
        return [ord(c) for c in text]

    def decode_ids(self, ids: List[int]) -> str:
        if self.ok:
            return self.tok.decode(ids, skip_special_tokens=True)
        return "".join(chr(i) for i in ids)

    def ntokens(self, text: str) -> int:
        return len(self.encode_ids(text))

def pad_ids_to(ids: List[int], target_len: int, pad_unit: List[int]) -> List[int]:
    out = list(ids)
    while len(out) < target_len:
        need = target_len - len(out)
        out.extend(pad_unit[:need] if len(pad_unit) >= need else pad_unit)
    return out[:target_len]

def make_corrupt_by_replacing_last_sentence(problem: str, tokwrap: TokWrap) -> str:
    """
    用第一句的 token 片段（长度与最后一句一致）替换“最后一句”，返回新的 problem 文本。
    """
    parts = split_preserve(problem)
    i_first = first_text_idx(parts)
    i_last = last_text_idx(parts)
    if i_first is None or i_last is None or i_first == i_last:
        return problem  # 结构不满足，直接返回

    first_text = parts[i_first].strip()
    last_text  = parts[i_last].strip()

    first_ids = tokwrap.encode_ids(first_text)
    last_ids  = tokwrap.encode_ids(last_text)
    L = len(last_ids)

    # pad 用 '?'
    pad_unit = tokwrap.encode_ids("?")
    if len(pad_unit) == 0:
        pad_unit = [ord('?')]

    # 取第一句的尾部 L 个 token；不足则先补 '?'
    if len(first_ids) < L:
        first_ids = pad_ids_to(first_ids, L, pad_unit)
        slice_ids = first_ids  # 刚好等长
    else:
        slice_ids = first_ids[-L:]

    repl = tokwrap.decode_ids(pad_ids_to(slice_ids, L, pad_unit)).strip()
    # 作为“问题”更自然：没有问号则加 '?'
    if not repl.endswith("?"):
        repl = repl.rstrip(".，,。") + "?"

    # 用替换后的最后一句回填
    # 尽量保持原空白风格
    prefix_ws = re.match(r'^\s*', parts[i_last]).group(0) or ""
    suffix_ws = re.match(r'.*?(\s*)$', parts[i_last]).group(1) or ""
    parts[i_last] = prefix_ws + repl + suffix_ws

    return "".join(parts)

def parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s)
    m = re.search(r'[-+]?((\d{1,3}(,\d{3})+)|\d+)(\.\d+)?([eE][-+]?\d+)?', s.replace("−", "-"))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except Exception:
        return None

def format_answer_str(x: Optional[float]) -> str:
    if x is None:
        return ""
    return str(int(x)) if float(x).is_integer() else str(float(x))

def make_wrong_answer(gold: Optional[float]) -> str:
    if gold is None:
        return ""
    if float(gold).is_integer():
        # 生成 ±1 作为干扰
        delta = random.choice([-1, 1])
        return str(int(gold) + delta)
    else:
        # 小数题：±10%
        return str(gold * (1.0 + random.choice([-0.1, 0.1])))

def build_item(template_id: str, problem: str, result: str, tokwrap: TokWrap) -> Dict[str, Any]:
    clean_prompt = f"{problem}\nAnswer with only the final numeric result.\nAnswer:"
    corrupt_problem = make_corrupt_by_replacing_last_sentence(problem, tokwrap)
    corrupt_prompt = f"{corrupt_problem}\nAnswer with only the final numeric result.\nAnswer:"

    gold = parse_number(result)
    ans = format_answer_str(gold)
    wrong = make_wrong_answer(gold)

    item = {
        "word_idxs": {  # 如需真实 span，可以再加：token 起止索引
            "start": 0,
            "end": 0
        },
        "prompts": [
            {
                "clean": clean_prompt,
                "corrupt": corrupt_prompt,
                "answers": [f" {ans}"] if ans != "" else [],
                "wrong_answers": [f" {wrong}"] if wrong != "" else [],
                "token_mismatch": False,  # 我们严格对齐，通常为 False
                "logic": str(template_id)  # 这里放 template_id；如需放模板描述可替换
            }
        ]
    }
    return item

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default="data/logic/math.json", help="输入 JSON: {tid: [{problem,result}, ...], ...}")
    ap.add_argument("--output", type=Path, default="data/corrupt/math.json", help="输出 JSON: List[...]（目标格式）")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                    help="用于长度判断的 tokenizer。失败则回退字符级。")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    data = json.loads(args.input.read_text(encoding="utf-8"))
    tokwrap = TokWrap(args.model_name)

    out: List[Dict[str, Any]] = []
    for tid, samples in data.items():
        for s in samples:
            prob = s.get("problem", "")
            res  = s.get("result", "")
            out.append(build_item(str(tid), prob, res, tokwrap))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(out)} items -> {args.output}")

if __name__ == "__main__":
    main()