#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circuit-attribution study (single-batch per logic):
• Discover salient circuit tags from a small training batch.
• Evaluate patched accuracy across all logic types.
• Export clean baseline and patched-accuracy matrix to CSV.

Author: Zhenyu Lei (refactored July 2025)
"""

from __future__ import annotations

# ────────────────────────────── Imports ──────────────────────────────
# std-lib
import argparse
import csv
import json
import logging
import math
import random
import re
from collections import defaultdict
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Set
import pdb

# third-party
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────── Logging ────────────────────────────────
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOG = logging.getLogger("CircuitMatrix")

# ───────────────────────────── Constants ─────────────────────────────
SUFFIX = " (Answer in True, False, or N/A (Neither)). Answer:"
_TAIL = re.compile(r"\s*\(answer in .*?answer:\s*$", re.I | re.S)

# ──────────────────────────── Utilities ─────────────────────────────
def ensure_suffix(text: str) -> str:
    return text if text.endswith(SUFFIX) else text + SUFFIX

def _align(src: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad / slice src to have seq_len == target_len (keeps grad)."""
    if src.size(1) == target_len:
        return src
    if src.size(1) == 1:                       # captured seq_len == 1
        return src.expand(-1, target_len, -1)
    if src.size(1) > target_len:               # captured longer
        return src[:, -target_len:, :]
    # captured shorter
    pad = src[:, -1:, :].expand(-1, target_len - src.size(1), -1)
    return torch.cat([src, pad], dim=1)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


POSSIBLE_ANSWERS: Set[str] = {"true", "false", "n/a", "na", "n", "neither"}

def normalise(raw: str) -> str:
    raw = raw.lower().strip().rstrip(".")
    if raw.startswith(("t", "y")):                     # true / yes
        return "true"
    if raw.startswith(("f")):                         # false
        return "false"
    # “n” “na” “n/a” “neither”
    if raw.startswith("n") or "neither" in raw:
        return "n/a"
    return "true" 


def _canon(prompt: str) -> str:
    prompt = _TAIL.sub("", prompt)
    prompt = prompt.replace(SUFFIX, "")
    return " ".join(prompt.split()).lower()


def first_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise AttributeError(names)


# ──────────────────────── Model helpers ─────────────────────────────


_LOGIT_VOCAB_CACHE: dict[str, list[int]] = {}   # canonical → list[token_id]

def _answer_token_ids(tok) -> dict[str, list[int]]:
    """
    把 {'true','false','n/a'} 映射到若干可能的 token-id，结果缓存。
    兼容 Qwen / GPT-NeoX 类分词器：' true' vs 'true' 都尝试。
    """
    global _LOGIT_VOCAB_CACHE
    if _LOGIT_VOCAB_CACHE:
        return _LOGIT_VOCAB_CACHE

    cand_map = {
        "true":  [" true", "true", " True"],
        "false": [" false", "false", " False"],
        "n/a":   [" n", "n", " na", "na", " n/a", "n/a",  # ← 多补几种单 token 写法
                  " neither", "neither", " N", "N", " N/A", "N/A"],
    }
    for canon, variants in cand_map.items():
        ids = []
        for v in variants:
            toks = tok(v, add_special_tokens=False).input_ids
            if len(toks) == 1:               # 只考虑单 token，速度快
                ids.append(toks[0])
        _LOGIT_VOCAB_CACHE[canon] = sorted(set(ids))
    return _LOGIT_VOCAB_CACHE


# ──────────────────────── 修改 generate_answer ───────────────────────
# def generate_answer(prompt: str, model, tok, src_path: Path, max_new: int = 5) -> str:
#     """
#     **新版**：一次 forward → 用 logits 直接判 ‘true/false/n/a’。
#     """
#     templ = prompt if "deductive" in src_path.name else f"{prompt}\n### The answer is:"
#     text  = ensure_suffix(templ)

#     if not text.endswith(" "):
#         text += " "

#     batch = tok(text, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         logits = model(**batch, use_cache=False).logits      # [1, L, V]
#     last = logits[:, -1, :]                                  # [1, V]

#     tok_ids = _answer_token_ids(tok)
#     scores = {
#         canon: max(last[0, ids]).item() if ids else float("-inf")
#         for canon, ids in tok_ids.items()
#     }
#     # 取分最高的 canonical label
#     best = max(scores.items(), key=lambda kv: kv[1])[0]
#     return best   # already 'true' / 'false' / 'n/a'
def generate_answer(prompt: str, model, tok, src_path: Path, max_new: int = 5) -> str:
    """Greedy-decode a short answer token and normalise it to {true,false,n/a}."""
    templ = prompt if "deductive" in src_path.name else f"{prompt}\n### The answer is:"
    model.eval()
    with torch.no_grad():
        ids = tok(ensure_suffix(templ), return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, use_cache=False)
        text = tok.decode(out[0], skip_special_tokens=True)

    # split after cue phrase
    after = text.split("Answer:")[-1] if "Answer:" in text else text.split("###")[-1]

    # ── robust to empty/whitespace outputs ─────────────────────────────
    words = after.strip().split()
    first = words[0] if words else ""          # empty string is OK
    return normalise(first)



# ───────────────── Attribution core (per-module scores) ─────────────


def module_scores(
    model,
    tok,
    pair: Dict,
    device,
    n_head: int,
) -> Dict[str, float]:
    """
    Compute |∂log p₍gold₎/∂h ⋅ (h_clean − h_corrupt)| averaged across
    batch & tokens, giving one scalar per head / module.
    """
    model.eval()
    batch = tok(
        [pair["clean"], pair["corrupt"]],
        return_tensors="pt",
        padding=True,
    ).to(device)
    ids, msk = batch["input_ids"], batch["attention_mask"]

    acts_c, acts_k, hooks = {}, {}, []

    def make_hook(buf):
        def fn(m, inp, out):
            tag = m._tag
            buf[tag] = inp[0] if tag.startswith("attn") else (
                out[0] if isinstance(out, tuple) else out
            )

        return fn

    def arm(buf):
        for i, blk in enumerate(model.model.layers):
            mods = {
                "ln": first_attr(blk, "input_layernorm", "ln_1"),
                "attn": blk.self_attn.o_proj,
                "mlp": blk.mlp,
            }
            for k, m in mods.items():
                m._tag = f"{k}_{i}"
                hooks.append(m.register_forward_hook(make_hook(buf)))

    # capture clean activations
    arm(acts_c)
    model(ids[:1], attention_mask=msk[:1], use_cache=False)
    for h in hooks:
        h.remove()
    hooks.clear()

    # capture corrupt & grad
    arm(acts_k)
    logits = model(ids[1:], attention_mask=msk[1:], use_cache=False).logits
    gold = torch.log_softmax(logits[:, -1], -1).max()
    grads = torch.autograd.grad(gold, list(acts_k.values()), retain_graph=False)

    out: Dict[str, float] = {}
    for g, tag in zip(grads, acts_k):
        hc, hk = acts_c[tag], acts_k[tag]
        if tag.startswith("attn"):
            head_dim = g.size(-1) // n_head
            g, hc, hk = (
                t.view(*t.shape[:-1], n_head, head_dim) for t in (g, hc, hk)
            )
            diff = (g * (hc - hk)).abs().mean(dim=(0, 1, 3))  # [n_head]
            out.update(
                {f"{tag}_h{h}": diff[h].item() for h in range(n_head)}
            )
        else:  # ln / mlp
            out[tag] = (g * (hc - hk)).abs().mean().item()
    return out


def avg_scores(
    model, tok, pairs: List[Dict], device, n_head: int
) -> Dict[str, float]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for p in pairs:
        for k, v in module_scores(model, tok, p, device, n_head).items():
            acc[k].append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def top_tags_single(scores: Dict[str, float], tau: float) -> Set[str]:
    k_top = max(1, math.ceil(tau * len(scores)))
    return {
        tag
        for tag, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[
            :k_top
        ]
    }


# ───────────────────────── Patch prediction ─────────────────────────


def patched_predict(
    model,
    tok,
    pair: Dict,
    tags: Set[str],
    device,
    src_path: Path,
) -> str:
    """
    1. 先对 clean prompt 前向一次，捕获选中 tag 的激活；
    2. 再对 corrupt prompt 前向 + 生成，同时把这些 tag 打补丁；
    3. 返回归一化后的答案字符串（'true'/'false'/'n/a'）。
    """
    store: Dict[str, torch.Tensor] = {}
    cap_hooks, rep_hooks = [], []
    # pdb.set_trace()
    # ---------- 工具：把 tag 映射到具体 module & (可选) head ----------
    def locate(tag: str):
        typ, layer, *rest = tag.split("_")          # e.g. attn_3_h5
        blk = model.model.layers[int(layer)]
        if typ == "ln":
            mod = first_attr(blk, "input_layernorm", "ln_1")
            return mod, None
        if typ == "mlp":
            return blk.mlp, None
        if typ == "attn":
            return blk.self_attn.o_proj, int(rest[0][1:])   # 'h5' → 5
        raise ValueError(tag)

    # ---------- 捕获 clean 激活 ----------
    def make_capturer(tag: str, head_idx: int | None):
        def fn(_m, inp, out):
            if head_idx is None:                      # ln / mlp
                act = out[0] if isinstance(out, tuple) else out
            else:                                     # attn head-wise
                n_head = model.config.num_attention_heads
                hd = inp[0].size(-1) // n_head
                act = (
                    inp[0]
                    .view(*inp[0].shape[:-1], n_head, hd)[:, :, head_idx, :]
                    .detach()
                )
            store[tag] = act.clone()
        return fn

    for tag in tags:
        mod, h = locate(tag)
        cap_hooks.append(mod.register_forward_hook(make_capturer(tag, h)))

    # 执行一次 clean 前向即可；不需要生成 token
    _ = model(
        **tok(ensure_suffix(pair["clean"]), return_tensors="pt").to(device),
        use_cache=False,
    )

    for h in cap_hooks:
        h.remove()
    cap_hooks.clear()

    # ---------- 打补丁并生成 corrupt 答案 ----------
    def make_replacer(tag: str, head_idx: int | None, mod):
        tgt = store[tag]            # captured on clean pass

        def fn(_m, inp, _out):
            # 对齐序列长度
            src = _align(tgt, inp[0].size(1))

            if head_idx is None:          # ---- ln / mlp 整块替换 ----
                return src

            # ---------- attn: 只替换一个 head ----------
            n_head = model.config.num_attention_heads
            hd     = inp[0].size(-1) // n_head
            x      = inp[0].view(*inp[0].shape[:-1], n_head, hd)
            x[..., head_idx, :] = src
            new_in = x.view_as(inp[0])
            return F.linear(new_in, _m.weight, _m.bias)

        return fn

    for tag in tags:
        mod, h = locate(tag)
        rep_hooks.append(mod.register_forward_hook(make_replacer(tag, h, mod)))

    # 用打补丁后的模型生成答案
    ans = generate_answer(pair["corrupt"], model, tok, src_path)

    # 清理 hook
    for h in rep_hooks:
        h.remove()

    return ans
    


def accuracy(pairs: List[Dict], pred_fn) -> float:
    """
    把两边都 canonicalise 到 {'true','false','n/a'} 再比较。
    """
    n_hit = 0
    for p in pairs:
        gold = normalise(p["answer"])
        pred = normalise(pred_fn(p))
        n_hit += int(gold == pred)
    return n_hit / len(pairs)


# ───────────────────────── Data handling ────────────────────────────


class Hub:
    """Lightweight loader that joins logic metadata with augmented pairs."""

    def __init__(self, logic_p: Path, aug_p: Path):
        self.logic_rows = self._load_logic(logic_p)
        self.aug_rows = self._load_aug(aug_p)
        self._attach_logic_labels()

    # private helpers -------------------------------------------------
    @staticmethod
    def _load_logic(p: Path) -> List[Dict]:
        rows = []
        for rec in json.loads(p.read_text()):
            logic_lbl = rec["question"][0]["<nl>"].strip()
            gold = str(rec["answer"]).lower()
            for cat, val in rec.items():
                if cat in {"question", "answer"} or not isinstance(val, dict):
                    continue
                for det in val.values():
                    prm = det.get("<nl>")
                    if prm:
                        rows.append(
                            {
                                "prompt": prm,
                                "gold": gold,
                                "logic": logic_lbl,
                                "canon": _canon(prm),
                            }
                        )
        return rows

    @staticmethod
    def _load_aug(p: Path) -> List[Dict]:
        rows = []
        for block in json.loads(p.read_text()):
            for prm in block["prompts"]:
                rows.append(
                    {
                        "clean": prm["clean"].strip(),
                        "corrupt": prm["corrupt"].strip(),
                        "answer": prm["answers"][0].strip(),
                        "canon": _canon(prm["clean"]),
                    }
                )
        return rows

    def _attach_logic_labels(self) -> None:
        canon_to_logic = {r["canon"]: r["logic"] for r in self.logic_rows}
        self.aug_rows = [
            r | {"logic": canon_to_logic[r["canon"]]}
            for r in self.aug_rows
            if r["canon"] in canon_to_logic
        ]

    # public helpers --------------------------------------------------
    def logics(self) -> List[str]:
        return sorted({r["logic"] for r in self.aug_rows})

    def sample_by_logic(self, logic: str, k: int) -> List[Dict]:
        pool = [r for r in self.aug_rows if r["logic"] == logic]
        return random.sample(pool, min(k, len(pool)))


# ────────────────────── Matrix computation class ────────────────────


class CircuitMatrixBuilder:
    """Discover circuits (one batch per logic) and evaluate patch accuracy."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, padding_side="left"
        )
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.n_head = self.model.config.num_attention_heads
        self.hub = Hub(args.deductive_json, args.augmented_json)
        self.logic_types = self.hub.logics()
        self.eval_sets = {
            lg: self.hub.sample_by_logic(lg, args.n_eval) for lg in self.logic_types
        }

    # -----------------------------------------------------------------
    def _clean_baseline(self):
        hit_sets = {}                         # logic → List[Dict]  (命中的 eval 样例)
        fn = lambda p: generate_answer(
            p["clean"], self.model, self.tok, self.args.deductive_json)
        accs = []
        for lg in self.logic_types:
            hits = [p for p in self.eval_sets[lg] if p["answer"].strip().lower() in fn(p).strip().lower()]
            hit_sets[lg] = hits
            accs.append(len(hits) / len(self.eval_sets[lg]))
        self.hit_sets = hit_sets             # ← 存一份
        return accs
    # def _clean_baseline(self) -> List[float]:
    #     fn = lambda p: generate_answer(
    #         p["clean"], self.model, self.tok, self.args.deductive_json
    #     )
    #     return [accuracy(self.eval_sets[lg], fn) for lg in self.logic_types]

    # -----------------------------------------------------------------
    def _patch_predict_fn(self, tags: Set[str]):
        return lambda p: patched_predict(
            self.model,
            self.tok,
            p,
            tags,
            self.device,
            self.args.deductive_json,
        )

    # -----------------------------------------------------------------
    def build(self):
        LOG.info("Building clean baseline …")
        baseline = self._clean_baseline()
        # pdb.set_trace()

        n = len(self.logic_types)
        matrix = [[0.0] * n for _ in range(n)]

        for i, lg_i in enumerate(self.logic_types):
            LOG.info("Discovering circuit for logic '%s' …", lg_i)
            train_examples = self.hub.sample_by_logic(lg_i, self.args.train_batch)
            scores = avg_scores(
                self.model, self.tok, train_examples, self.device, self.n_head
            )
            tags = top_tags_single(scores, self.args.tau)
            LOG.info(" » selected %d tags", len(tags))

            p_patch = self._patch_predict_fn(tags)
            for j, lg_j in enumerate(self.logic_types):
                matrix[i][j] = accuracy(self.hit_sets[lg_j], p_patch)
            # pdb.set_trace()

        return baseline, matrix

    # -----------------------------------------------------------------
    def save_csv(self, baseline, matrix):
        mdl_tag = self.args.model_name.replace("/", "_")
        out_mat = f"results_1batch_{mdl_tag}.csv"
        out_base = f"baseline_1batch_{mdl_tag}.csv"

        # accuracy matrix
        with open(out_mat, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow([""] + self.logic_types)
            for lg, row in zip(self.logic_types, matrix):
                w.writerow([lg] + [f"{x:.4f}" for x in row])

        # clean baseline
        with open(out_base, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["logic", "clean_acc"])
            for lg, acc in zip(self.logic_types, baseline):
                w.writerow([lg, f"{acc:.4f}"])

        LOG.info("✓ Matrix saved to %s", out_mat)
        LOG.info("✓ Baseline saved to %s", out_base)


# ────────────────────────────── CLI ────────────────────────────────


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("matrix-1batch")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument(
        "--deductive_json",
        type=Path,
        default=Path("data/logic/deductive_logic.json"),
    )
    p.add_argument(
        "--augmented_json",
        type=Path,
        default=Path("data/corrupt/augmented_dataset.json"),
    )
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument(
        "--train_batch",
        type=int,
        default=8,
        help="examples per logic for circuit discovery",
    )
    p.add_argument(
        "--n_eval", type=int, default=60, help="eval examples per logic"
    )
    p.add_argument("--seed", type=int, default=42)
    return p


# ──────────────────────────── Entrypoint ────────────────────────────


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)

    builder = CircuitMatrixBuilder(args)
    builder.model.config.use_cache = False
    baseline, matrix = builder.build()

    # pretty print
    print("\n=== Clean Baseline ===")
    for lg, acc in zip(builder.logic_types, baseline):
        print(f"{lg:<18}: {acc:.3f}")

    print("\n=== Patched Accuracy Matrix ===")
    head = " ".join(f"{lg[:4]:>6}" for lg in builder.logic_types)
    print(f"{'':8}{head}")
    for lg, row in zip(builder.logic_types, matrix):
        cells = " ".join(f"{x:6.3f}" for x in row)
        print(f"{lg[:8]:8}{cells}")

    builder.save_csv(baseline, matrix)


if __name__ == "__main__":
    main()