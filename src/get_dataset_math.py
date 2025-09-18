# src/get_dataset.py
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import random
from itertools import combinations

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer  # 仅类型提示可选

# ---------------- 基础工具 ----------------
def chunk_indices(n: int, bs: int) -> List[List[int]]:
    return [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]

# ---------------- 读取你“目标格式”的 JSON ----------------
def load_augmented_json_grouped(path: Path) -> Dict[str, List[Dict]]:
    """
    输入（你的目标格式，列表）:
    [
      {
        "word_idxs": {...},
        "prompts": [
          {
            "clean": "...",
            "corrupt": "...",
            "answers": [" 357"],        # 注意可能有前导空格
            "wrong_answers": [" 356"],  # 同理
            "token_mismatch": false,
            "logic": "0"
          }
        ]
      },
      ...
    ]

    输出:
      { logic_str: [ {clean, corrupt, answer, wrong_answer}, ... ] }
    """
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    blocks = json.loads(path.read_text(encoding="utf-8"))
    for block in blocks:
        for prm in block.get("prompts", []):
            logic = str(prm.get("logic", "")).strip()
            clean = prm.get("clean", "").strip()
            corrupt = prm.get("corrupt", "").strip()
            ans = ""
            if prm.get("answers"):
                ans = str(prm["answers"][0]).strip()
            wrong = ""
            if prm.get("wrong_answers"):
                wrong = str(prm["wrong_answers"][0]).strip()

            grouped[logic].append({
                "clean": clean,
                "corrupt": corrupt,
                "answer": ans,
                "wrong_answer": wrong,
            })
    return grouped

# ---------------- Dataset：与“第二段”一致 ----------------
class LogicDataset(Dataset):
    """
    __getitem__ 返回:
      { logic_str: [ [g1_dict, g2_dict], [g1_dict, g2_dict], ... ] }
    """
    def __init__(
        self,
        data: Dict[str, List[Dict]],
        tokenizer,                   # transformers.AutoTokenizer
        group_size: int,
        n_logic_per_item: int,
        max_length: int = 512,
        seed: int = 42,
    ) -> None:
        self.data = data
        self.tok = tokenizer
        self.group_size = group_size
        self.n_logic = n_logic_per_item
        self.max_length = max_length
        self.seed = seed

        self.logic_list = list(self.data.keys())
        assert len(self.logic_list) >= self.n_logic, "n_logic_per_item > total number of logics"
        self.index_set: List[Tuple[int, ...]] = list(combinations(range(len(self.logic_list)), self.n_logic))

        self.groups: Dict[str, List[List[int]]] = {
            lgc: chunk_indices(len(rows), self.group_size) for lgc, rows in self.data.items()
        }

        # 预编码
        self.clean_ids: Dict[str, torch.Tensor]   = {}
        self.clean_mask: Dict[str, torch.Tensor]  = {}
        self.corrupt_ids: Dict[str, torch.Tensor] = {}
        self.corrupt_mask: Dict[str, torch.Tensor]= {}

        for lgc, rows in self.data.items():
            clean_texts   = [r["clean"]   for r in rows]
            corrupt_texts = [r["corrupt"] for r in rows]

            enc_c = self.tok(
                clean_texts, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            enc_k = self.tok(
                corrupt_texts, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )

            self.clean_ids[lgc]    = enc_c["input_ids"]
            self.clean_mask[lgc]   = enc_c["attention_mask"]
            self.corrupt_ids[lgc]  = enc_k["input_ids"]
            self.corrupt_mask[lgc] = enc_k["attention_mask"]

    def __len__(self) -> int:
        return len(self.index_set)

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + idx)
        logic_idxs = self.index_set[idx]
        selected_logics = [self.logic_list[i] for i in logic_idxs]

        out_per_logic: Dict[str, List[List[Dict[str, torch.Tensor]]]] = {}
        for lgc in selected_logics:
            g_list = self.groups[lgc]
            if len(g_list) < 2:
                raise ValueError(f"Logic '{lgc}' needs >= 2 groups; reduce group_size or add data.")

            g1_idx, g2_idx = rng.sample(range(len(g_list)), 2)
            idxs1, idxs2 = g_list[g1_idx], g_list[g2_idx]
            m = min(len(idxs1), len(idxs2))
            idxs1, idxs2 = idxs1[:m], idxs2[:m]

            g1_dict = {
                "clean_ids":     self.clean_ids[lgc][idxs1],
                "clean_mask":    self.clean_mask[lgc][idxs1],
                "corrupt_ids":   self.corrupt_ids[lgc][idxs1],
                "corrupt_mask":  self.corrupt_mask[lgc][idxs1],
                # 重要：答案是字符串（数值），提供给 AP.calculate_effect 使用
                "answers_clean":  [self.data[lgc][i]["answer"]       for i in idxs1],
                "answers_corrupt":[self.data[lgc][i]["wrong_answer"] for i in idxs1],
            }
            g2_dict = {
                "clean_ids":     self.clean_ids[lgc][idxs2],
                "clean_mask":    self.clean_mask[lgc][idxs2],
                "corrupt_ids":   self.corrupt_ids[lgc][idxs2],
                "corrupt_mask":  self.corrupt_mask[lgc][idxs2],
                "answers_clean":  [self.data[lgc][i]["answer"]       for i in idxs2],
                "answers_corrupt":[self.data[lgc][i]["wrong_answer"] for i in idxs2],
            }

            out_per_logic.setdefault(lgc, []).append([g1_dict, g2_dict])

        return out_per_logic

# ---------------- collate：与“第二段”一致 ----------------
from collections import defaultdict as _dd
def collate_fn(batch):
    merged: Dict[str, List[List[Dict[str, torch.Tensor]]]] = _dd(list)
    for item in batch:
        for lgc, pair_list in item.items():
            merged[lgc].extend(pair_list)
    return merged