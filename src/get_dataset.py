#!/usr/bin/env python3
from __future__ import annotations

###############################################################################
# Standard library imports
###############################################################################
import argparse
import json
import logging
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import pdb
from typing import Dict, List, Tuple

###############################################################################
# Third-party imports
###############################################################################
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

###############################################################################
# Torch backend configuration (<=2.0 API style)
###############################################################################
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

###############################################################################
# Logging
###############################################################################
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(
    "flash: %s  • mem-eff: %s  • math: %s",
    torch.backends.cuda.flash_sdp_enabled(),
    torch.backends.cuda.mem_efficient_sdp_enabled(),
    torch.backends.cuda.math_sdp_enabled(),
)

SEED_DEFAULT = 42


###############################################################################
# Helper utilities
###############################################################################
def set_seed(seed: int = SEED_DEFAULT) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chunk_indices(n: int, bs: int) -> List[List[int]]:
    """Split range(n) into groups of length <= bs."""
    return [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]


###############################################################################
# Data loading (grouped by logic)
###############################################################################
def load_augmented_json_grouped(path: Path) -> Dict[str, List[Dict]]:
    """
    Returns:
        {logic_str: [ {clean, corrupt, answer, wrong_answer}, ... ] }
    """
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    with path.open() as fp:
        for block in json.load(fp):
            for prm in block.get("prompts", []):
                logic = prm.get("logic", "").strip()
                grouped[logic].append({
                    "clean": prm["clean"].strip(),
                    "corrupt": prm["corrupt"].strip(),
                    "answer": prm.get("answers", [""])[0].strip(),
                    "wrong_answer": prm.get("wrong_answers", [""])[0].strip(),
                })
    return grouped


###############################################################################
# Dataset
###############################################################################
class LogicDataset(Dataset):
    """
    __getitem__ 返回:
      { logic_str: [ [g1_dict, g2_dict], [g1_dict, g2_dict], ... ] }
    其中 g1_dict/g2_dict 结构相同，包含 tokenized tensors 与 answer 列表。
    """

    def __init__(
        self,
        data: Dict[str, List[Dict]],
        tokenizer: AutoTokenizer,
        group_size: int,
        n_logic_per_item: int,
        max_length: int = 512,
        seed: int = SEED_DEFAULT,
    ) -> None:
        self.data               = data
        self.tok                = tokenizer
        self.group_size         = group_size
        self.n_logic            = n_logic_per_item
        self.max_length         = max_length
        self.seed               = seed

        self.logic_list         = list(self.data.keys())
        assert len(self.logic_list) >= self.n_logic, "n_logic_per_item > total number of logics"
        self.index_set: List[Tuple[int, ...]] = list(combinations(range(len(self.logic_list)), self.n_logic))

        # groups per logic
        self.groups: Dict[str, List[List[int]]] = {
            lgc: chunk_indices(len(rows), self.group_size) for lgc, rows in self.data.items()
        }

        # pre-tokenize
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

    def __getitem__(self, idx: int) -> Dict[str, List[List[Dict[str, torch.Tensor]]]]:
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
                "answers_clean":  [self.data[lgc][i]["answer"]        for i in idxs1],
                "answers_corrupt":[self.data[lgc][i]["wrong_answer"]  for i in idxs1],
            }
            g2_dict = {
                "clean_ids":     self.clean_ids[lgc][idxs2],
                "clean_mask":    self.clean_mask[lgc][idxs2],
                "corrupt_ids":   self.corrupt_ids[lgc][idxs2],
                "corrupt_mask":  self.corrupt_mask[lgc][idxs2],
                "answers_clean":  [self.data[lgc][i]["answer"]        for i in idxs2],
                "answers_corrupt":[self.data[lgc][i]["wrong_answer"]  for i in idxs2],
            }

            out_per_logic.setdefault(lgc, []).append([g1_dict, g2_dict])

        return out_per_logic


###############################################################################
# ---------------- Protected set built from LogicDataset groups ----------------
from torch.utils.data import Dataset, DataLoader
import random

class ProtectedDataset(Dataset):
    """
    Builds a small set of (clean prompt, correct answer) pairs from grouped rows:
        grouped_rows: { logic_str: [ {clean, corrupt, answer, wrong_answer}, ... ] }
    Returns per item:
        {
          "input_ids":      LongTensor [seq],
          "attention_mask": LongTensor [seq],
          "target_ids":     LongTensor [seq]  # last position holds the target token id
        }
    """
    def __init__(
        self,
        grouped_rows: Dict[str, List[Dict]],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        n_logics: int = 16,          # how many distinct logics to sample
        per_logic: int = 2,          # how many examples per selected logic
        seed: int = 0,
    ):
        super().__init__()
        self.tok = tokenizer
        self.max_length = max_length

        rng = random.Random(seed)
        all_logics = list(grouped_rows.keys())
        rng.shuffle(all_logics)
        picked = all_logics[:min(n_logics, len(all_logics))]

        # Collect clean prompts + correct answer
        pairs = []
        for lgc in picked:
            rows = grouped_rows[lgc]
            if not rows:
                continue
            idxs = list(range(len(rows)))
            rng.shuffle(idxs)
            for i in idxs[:per_logic]:
                r = rows[i]
                pairs.append((r["clean"], r.get("answer", "")))

        # Tokenize prompts
        enc = self.tok(
            [p for p, _ in pairs],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Build target_ids: a sequence filled with pad, except the *last* position
        # stores the first token id of the answer string (like your token_logit_metric).
        tgt_ids = []
        for _, ans in pairs:
            ids = self.tok.encode(ans, add_special_tokens=False)
            if len(ids) == 0:
                ids = [self.tok.eos_token_id]
            tgt_ids.append(ids[0])

        self.input_ids = enc["input_ids"]                # [N, L]
        self.attn_mask = enc["attention_mask"]           # [N, L]
        N, L = self.input_ids.size()
        self.target_ids = torch.full((N, L), fill_value=self.tok.pad_token_id, dtype=torch.long)
        self.target_ids[:, -1] = torch.tensor(tgt_ids, dtype=torch.long)  # gather from last logit

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx: int):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "target_ids":     self.target_ids[idx],
        }


def build_protected_loader(
    grouped_rows: Dict[str, List[Dict]],
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    n_logics: int = 16,
    per_logic: int = 2,
    seed: int = 0,
    batch_size: int = 4,
    shuffle: bool = True,
):
    ds = ProtectedDataset(
        grouped_rows,
        tokenizer=tokenizer,
        max_length=max_length,
        n_logics=n_logics,
        per_logic=per_logic,
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return ds, loader


###############################################################################
# Collate
###############################################################################
def collate_fn(
    batch: List[Dict[str, List[List[Dict[str, torch.Tensor]]]]]
) -> Dict[str, List[List[Dict[str, torch.Tensor]]]]:
    """
    Merge logic keys, simply extend the pair lists.
    group_size>1 时，逻辑相同的 pair 会累加。
    输出结构保持与 __getitem__ 相同：
        { logic: [ [g1_dict, g2_dict], [g1_dict, g2_dict], ... ] }
    """
    merged: Dict[str, List[List[Dict[str, torch.Tensor]]]] = defaultdict(list)
    for item in batch:
        for lgc, pair_list in item.items():
            merged[lgc].extend(pair_list)
    return merged


###############################################################################
# Main
###############################################################################
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augmented-only LogicDataset pipeline (grouped by logic)")
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--augmented_json", type=Path, default=Path("data/corrupt/augmented_dataset.json"))
    p.add_argument("--group_size", type=int, default=8)           # samples per group inside one logic
    p.add_argument("--n_logic_per_item", type=int, default=3)     # how many logics per dataset item
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--batch_size", type=int, default=1)            # DataLoader batch_size
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    grouped_rows = load_augmented_json_grouped(args.augmented_json)

    dataset = LogicDataset(
        data=grouped_rows,
        tokenizer=tok,
        group_size=args.group_size,
        n_logic_per_item=args.n_logic_per_item,
        max_length=args.max_len,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # ---- sanity check ----
    batch = next(iter(loader))
    first_logic, pair_list = next(iter(batch.items()))
    logger.info("Logic key preview: %s...", first_logic.splitlines()[0][:60])
    logger.info("#pairs for this logic in this batch: %d", len(pair_list))
    g1_dict, g2_dict = pair_list[0]
    logger.info("g1 clean_ids shape: %s", g1_dict["clean_ids"].shape)
    logger.info("g2 clean_ids shape: %s", g2_dict["clean_ids"].shape)

    # ---- forward example ----
    with torch.no_grad():
        logits = model(
            input_ids=g1_dict["clean_ids"].to(device),
            attention_mask=g1_dict["clean_mask"].to(device),
        ).logits
        _ = logits[:, -1, :]

    logger.info("Done.")


if __name__ == "__main__":
    main()