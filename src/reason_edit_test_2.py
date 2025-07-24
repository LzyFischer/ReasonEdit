#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circuit-attribution study (single batch per logic) using **Edge Attribution Patching
with Integrated Gradients on inputs (EAP-IG-inputs)**, where the **edge attribution
scores remain in the computation graph and can receive / propagate gradients**.

Key changes vs. your previous script:
  • Use a differentiable attribution routine (`attribute_diff`) that returns a
    `scores` tensor with `requires_grad=True` (no `.detach()` / in-place buffers).
  • Separate metrics: one differentiable metric for attribution (log-prob loss),
    one non-differentiable accuracy metric for reporting.
  • Avoid detaching scores in `eap_ig_scores` (renamed to `eap_ig_scores_diff`).

After you obtain `scores`, you can build any loss on top of it (e.g. L1, sparsity,
regularizers) and run `loss.backward()` to backprop through both the model and the
attribution pipeline (if you keep graphs).

Author: Zhenyu Lei (EAP-IG grad-friendly refactor, July 2025)
"""
from __future__ import annotations

# ────────────────────────────── Imports ──────────────────────────────
import argparse
import csv
import json
import logging
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

# EAP package (assumed to be installed / on PYTHONPATH)
from src.eap.graph import Graph
from src.eap.attr_diff import attribute_diff   # ← differentiable EAP-IG implementation
from src.eap.evaluate import evaluate_graph, evaluate_baseline

# ──────────────────────────── Logging ────────────────────────────────
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOG = logging.getLogger("CircuitMatrixEAPIG-Grad")

# ───────────────────────────── Utilities ─────────────────────────────
POSSIBLE_ANSWERS: Set[str] = {"true", "false", "n/a"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Canonicalise answers ------------------------------------
import re
SUFFIX = " (Answer in True, False, or N/A (Neither)). Answer:"
_TAIL = re.compile(r"\s*\(answer in .*?answer:\s*$", re.I | re.S)


def normalise(raw: str) -> str:
    raw = raw.lower().strip().rstrip(".")
    if raw.startswith(("t", "y")):
        return "true"
    if raw.startswith("f"):
        return "false"
    if raw.startswith("n") or "neither" in raw:
        return "n/a"
    return "true"


def _canon(prompt: str) -> str:
    prompt = _TAIL.sub("", prompt)
    prompt = prompt.replace(SUFFIX, "")
    return " ".join(prompt.split()).lower()


# ---------- Answer token id map (single-token only for speed) --------
_LOGIT_VOCAB_CACHE: Dict[str, List[int]] = {}


def _answer_token_ids(tok: AutoTokenizer) -> Dict[str, List[int]]:
    global _LOGIT_VOCAB_CACHE
    if _LOGIT_VOCAB_CACHE:
        return _LOGIT_VOCAB_CACHE
    cand_map = {
        "true":  [" true", "true", " True"],
        "false": [" false", "false", " False"],
        "n/a":   [" n", "n", " na", "na", " n/a", "n/a", " neither", "neither", " N", "N", " N/A", "N/A"],
    }
    for canon, variants in cand_map.items():
        ids = []
        for v in variants:
            toks = tok(v, add_special_tokens=False).input_ids
            if len(toks) == 1:
                ids.append(toks[0])
        _LOGIT_VOCAB_CACHE[canon] = sorted(set(ids))
    return _LOGIT_VOCAB_CACHE


# ───────────────────────── Dataset & Metrics ─────────────────────────
class PairDataset(Dataset):
    """(clean, corrupt, label) triples for attribution / evaluation."""

    def __init__(self, pairs: List[Dict]):
        self.clean = [p["clean"] for p in pairs]
        self.corrupt = [p["corrupt"] for p in pairs]
        self.label = [p["answer"] for p in pairs]

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.clean[idx], self.corrupt[idx], self.label[idx]


def make_acc_metric(tok_ids_map: Dict[str, List[int]]) -> Callable[[Tensor, Tensor, Tensor, List[str]], Tensor]:
    """Non-differentiable accuracy (0/1) per example. Use only for reporting."""

    def metric(logits: Tensor, _clean_logits: Tensor, lengths: Tensor, labels: List[str]) -> Tensor:
        last = logits[torch.arange(logits.size(0)), lengths - 1]
        logprobs = F.log_softmax(last, dim=-1)
        outs = []
        for i, lab in enumerate(labels):
            gold = normalise(lab)
            pred = max(
                POSSIBLE_ANSWERS,
                key=lambda k: logprobs[i, tok_ids_map[k]].logsumexp(0),
            )
            outs.append(float(pred == gold))
        return torch.tensor(outs, device=logits.device, dtype=logits.dtype)

    return metric


def make_logprob_metric(tok_ids_map: Dict[str, List[int]]) -> Callable[[Tensor, Tensor, Tensor, List[str]], Tensor]:
    """Differentiable metric: negative log-prob of the correct answer token-set.
    Returns mean over batch (scalar) or per-example vector if you prefer.
    We'll return per-example vector to match EAP code expectations.
    """

    def metric(logits: Tensor, _clean_logits: Tensor, lengths: Tensor, labels: List[str]) -> Tensor:
        last = logits[torch.arange(logits.size(0)), lengths - 1]  # [B, V]
        logprobs = F.log_softmax(last, dim=-1)  # [B,V]
        losses = []
        for i, lab in enumerate(labels):
            gold = normalise(lab)
            ids = tok_ids_map[gold]
            # -log(sum p(ids)) (logsumexp then negative)
            nll = -torch.logsumexp(logprobs[i, ids], dim=0)
            losses.append(nll)
        return torch.stack(losses)  # [B]

    return metric


# ───────────────────────── EAP-IG helpers ────────────────────────────
@dataclass
class EdgeScores:
    scores: torch.Tensor  # [n_forward, n_backward], requires_grad=True
    graph: Graph


def eap_ig_scores_diff(
    model: HookedTransformer,
    graph: Graph,
    ds: Dataset,
    metric_diff: Callable,
    ig_steps: int = 30,
    batch_size: int = 8,
    quiet: bool = False,
) -> EdgeScores:
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    # attribute_diff returns a tensor with grad
    scores = attribute_diff(
        model,
        graph,
        dl,
        metric_diff,
        method="EAP-IG-inputs",
        ig_steps=ig_steps,
        quiet=quiet,
    )
    return EdgeScores(scores=scores, graph=graph)


def top_edges(edge_scores: EdgeScores, tau: float) -> Graph:
    """Keep top-τ fraction edges by |score|. This is non-differentiable (selection)."""
    scores = edge_scores.scores
    g = edge_scores.graph
    flat = scores.abs().view(-1)
    k = max(1, int(tau * flat.numel()))
    _, idx = torch.topk(flat, k)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    mask = mask.view_as(scores)

    # reset & set
    for e in g.edges.values():
        e.in_graph = False
    for e in g.edges.values():
        f = g.forward_index(e.parent, attn_slice=False)
        b = g.backward_index(e.child, qkv=getattr(e, "qkv", None), attn_slice=False)
        if mask[f, b]:
            e.in_graph = True
    return g


# ───────────────────────── Data handling ────────────────────────────
class Hub:
    """Load logic metadata and augmented clean/corrupt pairs, attach logic labels."""

    def __init__(self, logic_p: Path, aug_p: Path):
        self.logic_rows = self._load_logic(logic_p)
        self.aug_rows = self._load_aug(aug_p)
        self._attach_logic_labels()

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
                        rows.append({
                            "prompt": prm,
                            "gold": gold,
                            "logic": logic_lbl,
                            "canon": _canon(prm),
                        })
        return rows

    @staticmethod
    def _load_aug(p: Path) -> List[Dict]:
        rows = []
        for block in json.loads(p.read_text()):
            for prm in block["prompts"]:
                rows.append({
                    "clean": prm["clean"].strip(),
                    "corrupt": prm["corrupt"].strip(),
                    "answer": prm["answers"][0].strip(),
                    "canon": _canon(prm["clean"]),
                })
        return rows

    def _attach_logic_labels(self) -> None:
        canon_to_logic = {r["canon"]: r["logic"] for r in self.logic_rows}
        self.aug_rows = [
            r | {"logic": canon_to_logic[r["canon"]]}
            for r in self.aug_rows
            if r["canon"] in canon_to_logic
        ]

    def logics(self) -> List[str]:
        return sorted({r["logic"] for r in self.aug_rows})

    def sample_by_logic(self, logic: str, k: int) -> List[Dict]:
        pool = [r for r in self.aug_rows if r["logic"] == logic]
        return random.sample(pool, min(k, len(pool)))


# ────────────────────── Matrix computation class ────────────────────
class CircuitMatrixBuilder:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer for answer-token mapping
        self.tok = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, padding_side="left"
        )
        self.tok.pad_token = self.tok.eos_token

        # HookedTransformer
        self.model = HookedTransformer.from_pretrained(
            args.model_name,
            device=str(self.device),
            dtype=torch.float32,
            fold_ln=False,
            center_writing_weights=False,
            # use_attn_result=True,
            # use_split_qkv_input=True,
            # use_hook_mlp_in=True,
            # ungroup_grouped_query_attention=True,
        )
        self.model.set_use_attn_result(True)
        self.model.set_use_split_qkv_input(True)
        self.model.set_use_hook_mlp_in(True)
        self.model.set_ungroup_grouped_query_attention(True)
        self.model.cfg.use_cache = False

        self.graph = Graph.from_model(self.model)
        self.tok_ids_map = _answer_token_ids(self.tok)

        # Metrics
        self.metric_eval = make_acc_metric(self.tok_ids_map)        # for printing accuracy
        self.metric_diff = make_logprob_metric(self.tok_ids_map)    # for attribution (differentiable)

        # Data hub
        self.hub = Hub(args.deductive_json, args.augmented_json)
        self.logic_types = self.hub.logics()

        # Eval sets
        self.eval_sets: Dict[str, List[Dict]] = {
            lg: self.hub.sample_by_logic(lg, args.n_eval) for lg in self.logic_types
        }

    # -----------------------------------------------------------------
    def _clean_baseline(self) -> List[float]:
        accs = []
        for lg in self.logic_types:
            ds = PairDataset(self.eval_sets[lg])
            dl = DataLoader(ds, batch_size=self.args.batch_size, shuffle=False)
            base = evaluate_baseline(self.model, dl, self.metric_eval).mean().item()
            accs.append(base)
        return accs

    # -----------------------------------------------------------------
    def build(self) -> Tuple[List[float], List[List[float]]]:
        LOG.info("Building clean baseline …")
        # baseline = self._clean_baseline()

        n = len(self.logic_types)
        matrix = [[0.0] * n for _ in range(n)]

        for i, lg_i in enumerate(self.logic_types):
            LOG.info("Discovering circuit (EAP-IG diff) for logic '%s' …", lg_i)
            train_examples = self.hub.sample_by_logic(lg_i, self.args.train_batch)

            # ---- Differentiable edge attribution ----
            g_tmp = Graph.from_model(self.model)   # 或 Graph.from_model(self.model.cfg)
            es = eap_ig_scores_diff(
                self.model,
                g_tmp,
                PairDataset(train_examples),
                self.metric_diff,
                ig_steps=self.args.ig_steps,
                batch_size=self.args.batch_size,
                quiet=True,
            )

            # You can build any differentiable loss here (optional):
            # e.g., sparsity_loss = es.scores.abs().mean(); sparsity_loss.backward()
            # For now, just select top-τ (non-differentiable step):
            g_kept = top_edges(es, self.args.tau)
            kept_cnt = sum(e.in_graph for e in g_kept.edges.values())
            LOG.info(" » selected %d edges", kept_cnt)

            # ---- Evaluate patched accuracy across all logics ----
            for j, lg_j in enumerate(self.logic_types):
                ds_eval = PairDataset(self.eval_sets[lg_j])
                dl_eval = DataLoader(ds_eval, batch_size=self.args.batch_size, shuffle=False)
                perf = (
                    evaluate_graph(
                        self.model,
                        g_kept,
                        dl_eval,
                        self.metric_eval,
                        intervention="patching",
                        quiet=True,
                        skip_clean=False,
                    )
                    .mean()
                    .item()
                )
                matrix[i][j] = perf

        return baseline, matrix

    # -----------------------------------------------------------------
    def save_csv(self, baseline: List[float], matrix: List[List[float]]):
        mdl_tag = self.args.model_name.replace("/", "_")
        out_mat = f"results_1batch_{mdl_tag}.csv"
        out_base = f"baseline_1batch_{mdl_tag}.csv"

        with open(out_mat, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow([""] + self.logic_types)
            for lg, row in zip(self.logic_types, matrix):
                w.writerow([lg] + [f"{x:.4f}" for x in row])

        with open(out_base, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["logic", "clean_acc"])
            for lg, acc in zip(self.logic_types, baseline):
                w.writerow([lg, f"{acc:.4f}"])

        LOG.info("✓ Matrix saved to %s", out_mat)
        LOG.info("✓ Baseline saved to %s", out_base)


# ────────────────────────────── CLI ────────────────────────────────

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("matrix-1batch-eap-ig-grad")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--deductive_json", type=Path, default=Path("data/logic/deductive_logic.json"))
    p.add_argument("--augmented_json", type=Path, default=Path("data/corrupt/augmented_dataset.json"))
    p.add_argument("--tau", type=float, default=0.1, help="fraction of edges kept")
    p.add_argument("--train_batch", type=int, default=8, help="examples per logic for edge discovery")
    p.add_argument("--n_eval", type=int, default=60, help="eval examples per logic")
    p.add_argument("--ig_steps", type=int, default=30, help="IG steps for EAP-IG")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p


# ──────────────────────────── Entrypoint ────────────────────────────

def main():
    args = get_parser().parse_args()
    set_seed(args.seed)

    builder = CircuitMatrixBuilder(args)
    baseline, matrix = builder.build()

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
