#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_attr_scores_eap.py
────────────────────────────────────────────────────────────────────────────
用 EAP（Edge Attribution Patching）替换 mask_gradient_prune_scores 流水线，
完全移除 auto_circuit 依赖，但保留原有数据集/切分/阈值与落盘结构。

输出（按你的原路径保持不变）：
- 稀疏边 JSON：outputs/attr_scores/<dataset>/<model>/<keep%>/<run_tag>/<logic_name>_split{seed}_part{A|B}.json
- 热力图 PNG：    .../figures/<logic_name>_{subset_tag}.png
- 电路图 PNG：    .../figures/<logic_name>_{subset_tag}_circuit_top100.png
"""

from __future__ import annotations

# ───────────────────────────── Imports ─────────────────────────────────────
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pdb

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import re

# 依赖：你上传的本地文件（与本脚本位于同一目录或加入 PYTHONPATH）
# - src/eap_wrapper.py: 提供 EAP(...) 主入口
# - src/eap_graph.py:   提供 EAPGraph 类与图结构
try:
    from src.eap_wrapper import EAP  # type: ignore
    from src.eap_graph import EAPGraph  # type: ignore
except Exception as _e:
    EAP = None
    EAPGraph = None
    _IMPORT_ERR = _e

# 直接使用 TransformerLens 的 HookedTransformer
from transformer_lens import HookedTransformer


# ───────────────────────────── Constants ───────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATA = "data/corrupt/level_1.json"
DEFAULT_OUT  = "outputs/attr_scores"
TOP_N_EDGES  = 200          # 电路图使用的 top-N 边
FIG_DPI      = 200
ATTN_EDGE_SCALE = 1.0      # ← 新增：attn 相关边的放大倍数
CAND_MULTIPLIER = 3  

# SDP 开关：与原脚本保持一致，避免 Flash/高效SDP引入的不确定性
os.environ["PYTORCH_SDP_BACKEND"] = "math"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ───────────────────────────── Utils ───────────────────────────────────────
def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_tag(name: str) -> str:
    last = name.split("/")[-1]
    return "".join("_" if c in ".-" else c for c in last).lower()


def quant_tag(q: float) -> str:
    """把分位 q 转成“保留百分比”的 tag，如 q=0.9 → '10'（保留 10%）"""
    keep_pct = round((1.0 - q) * 100, 3)
    return ("%g" % keep_pct).replace(".", "_")


def add_identity_mid_hooks(model: HookedTransformer) -> None:
    """为每个 block 补上 hook_resid_mid = nn.Identity()（命名兼容你的可视化）"""
    for i, block in enumerate(model.blocks):
        if not hasattr(block, "hook_resid_mid"):
            block.hook_resid_mid = nn.Identity()
            block.hook_resid_mid.name = f"blocks.{i}.hook_resid_mid"


def try_resume_state_dict(model: HookedTransformer, ckpt_path: Path, strict: bool) -> None:
    """尽量把 state_dict 灌进 TL 模型；缺失/多余 key 打印出来但不中断。"""
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[ckpt] loaded from {ckpt_path}")
    print(f"       missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print("       missing keys (head):", missing[:10])
    if unexpected:
        print("       unexpected keys (head):", unexpected[:10])
    pdb.set_trace()


def to_tokens_batched(model: HookedTransformer, texts: List[str]) -> torch.Tensor:
    """把一组文本 tokenization + 左侧 pad 到同长；加 BOS。"""
    return model.to_tokens(texts, prepend_bos=True)  # [B, S]


def make_avg_diff_metric(model: HookedTransformer, true_str: str, false_str: str):
    """
    avg_diff：取 batch 中每条样本末位置的 logit(True_last_tok) - logit(False_last_tok) ，再对 batch 求均值。
    """
    true_tok = model.to_tokens(true_str, prepend_bos=True)[0].tolist()[-1]
    false_tok = model.to_tokens(false_str, prepend_bos=True)[0].tolist()[-1]

    def metric(logits: torch.Tensor) -> torch.Tensor:
        last_pos = logits.size(1) - 1
        diff = logits[:, last_pos, true_tok] - logits[:, last_pos, false_tok]
        return diff.mean()

    return metric


def extract_clean_corrupt_pairs(examples: List[Dict]) -> Tuple[List[str], List[str]]:
    """从 logic_group 的 prompts 数组中提取 clean/corrupt 文本对；不规范时兜底复制。"""
    clean, corrupt = [], []
    for ex in examples:
        if isinstance(ex, dict) and ("clean" in ex) and ("corrupt" in ex):
            clean.append(ex["clean"])
            corrupt.append(ex["corrupt"])
        else:
            txt = ex if isinstance(ex, str) else json.dumps(ex, ensure_ascii=False)
            clean.append(txt)
            corrupt.append(txt)
    return clean, corrupt


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_heatmap(scores: torch.Tensor, out_png: Path, title: str = "") -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(scores.numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Downstream nodes")
    plt.ylabel("Upstream nodes")
    plt.tight_layout()
    plt.savefig(out_png, dpi=FIG_DPI)
    plt.close()


# ───────────────────────────── EAP Core ────────────────────────────────────
def eap_scores_for_subset(
    model: HookedTransformer,
    clean_texts: List[str],
    corrupt_texts: List[str],
    metric_true: str,
    metric_false: str,
    batch_size: int,
) -> "EAPGraph":
    """跑一遍 EAP，返回带有 eap_scores 的图对象（graph.eap_scores: [U, D]）"""
    if EAP is None or EAPGraph is None:
        raise RuntimeError(
            f"Failed to import EAP/EAPGraph from src/. Original error:\n{_IMPORT_ERR}"
        )

    clean_tokens = to_tokens_batched(model, clean_texts)
    corrupt_tokens = to_tokens_batched(model, corrupt_texts)

    metric = make_avg_diff_metric(model, metric_true, metric_false)

    # 开启必要中间钩子
    model.set_use_attn_result(True)
    # model.set_use_attn_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)

    graph = EAP(
        model=model,
        clean_tokens=clean_tokens,
        corrupted_tokens=corrupt_tokens,
        metric=metric,
        upstream_nodes=None,       # eap_graph 的默认节点集合
        downstream_nodes=None,     # eap_graph 的默认节点集合
        batch_size=batch_size,
    )
    return graph


def threshold_and_sparse(graph: "EAPGraph", q: float) -> Dict:
    """
    对 eap_scores 做全局分位数阈值，返回稀疏字典：
    {
      "shape": [U, D],
      "threshold": float,
      "edges": [ [u_idx, d_idx, score], ... ]   # 仅保留 >= threshold
    }
    """
    scores = graph.eap_scores  # Tensor[U, D] (cpu)
    flat = scores.flatten()
    thresh = torch.quantile(flat, q).item() if flat.numel() else 0.0
    keep_mask = scores >= thresh
    u_idx, d_idx = torch.nonzero(keep_mask, as_tuple=True)
    kept_vals = scores[keep_mask]
    edges = torch.stack([u_idx, d_idx, kept_vals], dim=1).tolist()
    return {
        "shape": [int(scores.size(0)), int(scores.size(1))],
        "threshold": float(thresh),
        "edges": edges,
    }


# ──────────────────────── Parsing & Graph Building ─────────────────────────
# 更宽松的 layer / head 提取（保留接口，必要时扩展）
_LAYERS = [re.compile(r"(?:blocks?|layer|blk)[\._\-\/]*(\d+)", re.I)]
_HEADS  = [
    re.compile(r"(?:\.|_|/|^|-)h(?:ead)?[\._\-\/]*(\d+)", re.I),
    re.compile(r"(?:heads?)[\._\-\/]*(\d+)", re.I),
]

def _parse_eap_node_relaxed(node_name: str) -> Tuple[int, str, int]:
    """
    兼容以下命名：
      - resid_pre.<L>        → (L, 'input', 0)
      - resid_post.<L>       → (L, 'output', 0)
      - mlp.<L>              → (L, 'down_proj', 0)
      - head.<L>.<H>.*       → (L, 'attn_head', H)
    解析失败兜底：layer=0, type='output', index=0
    """
    name = str(node_name)
    parts = name.split(".")

    if name.startswith("resid_pre."):
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0, "input", 0
    if name.startswith("resid_post."):
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0, "output", 0
    if name.startswith("mlp."):
        return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0, "down_proj", 0
    if name.startswith("head."):
        L = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        H = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        return L, "attn_head", H

    m = re.search(r"(\d+)", name)
    L = int(m.group(1)) if m else 0
    return L, "output", 0


def build_nx_from_top_edges(
    graph: "EAPGraph",
    n: int = TOP_N_EDGES,
    abs_scores: bool = True,
    include_attn: bool = True,
    include_mlp: bool = True,
) -> nx.DiGraph:
    """
    用 EAPGraph 的 top-n 边构建 nx.DiGraph。
    节点键为 "m,L,I,T"，原名存在属性 'raw' 中，以便贴标签。
    这里会对“含 attn 的边”做权重放大（ATTN_EDGE_SCALE）。
    """
    # 1) 先多取一些候选边（避免放大后没机会入选）
    raw_n = max(n * CAND_MULTIPLIER, n + 100)
    raw_edges = list(graph.top_edges(n=raw_n, abs_scores=abs_scores))

    boosted_edges = []
    for u_name, v_name, s in raw_edges:
        uL, uT, uI = _parse_eap_node_relaxed(u_name)
        vL, vT, vI = _parse_eap_node_relaxed(v_name)

        # 过滤（和原逻辑一致）
        if not include_attn and (uT in {"attn_head", "o_proj"} or vT in {"attn_head", "o_proj"}):
            continue
        if not include_mlp and (uT in {"mlp_in", "down_proj"} or vT in {"mlp_in", "down_proj"}):
            continue

        # 2) 如果任一端是 attn 相关，放大该边的分数
        is_attn_edge = (uT in {"attn_head", "o_proj"}) or (vT in {"attn_head", "o_proj"})
        s_boost = np.abs(float(s)) * (1.5 if is_attn_edge else 1.0) 
        boosted_edges.append((u_name, v_name, s_boost, uL, uT, uI, vL, vT, vI))

    # 3) 按放大后的绝对值重新排序，截取前 n
    boosted_edges.sort(key=lambda x: abs(x[2]), reverse=True)
    boosted_edges = boosted_edges[:n]

    # 4) 构建图（使用放大后的权重）
    G = nx.DiGraph()
    def key(L: int, T: str, I: int) -> str:
        return f"m,{L},{I},{T}"

    for (u_name, v_name, s_boost, uL, uT, uI, vL, vT, vI) in boosted_edges:
        if uL < 0 or vL < 0:
            continue
        ukey, vkey = key(uL, uT, uI), key(vL, vT, vI)
        if ukey not in G:
            G.add_node(ukey, raw=str(u_name), layer=int(uL), typ=uT, idx=int(uI))
        if vkey not in G:
            G.add_node(vkey, raw=str(v_name), layer=int(vL), typ=vT, idx=int(vI))
        G.add_edge(ukey, vkey, weight=float(s_boost))

    return G


def _prune_keep_bidir_with_context(G: nx.DiGraph) -> nx.DiGraph:
    """
    只保留：既有入边也有出边的节点（核心节点）以及这些核心节点的所有直接前驱/后继。
    若筛完为空，则回退为原图以避免空图崩溃。
    """
    core = {n for n in G.nodes() if G.in_degree(n) > 0 and G.out_degree(n) > 0}
    keep = set(core)
    for n in list(core):
        keep.update(G.predecessors(n))
        keep.update(G.successors(n))
    if not keep:
        return G
    H = G.subgraph(keep).copy()
    return H


# ───────────────────────── Visualization (Graphviz) ────────────────────────
def _node_style_for_typ(typ: str) -> Tuple[str, str]:
    palette = {
        "input":     ("#d7f0d0", "#6ca966"),
        "attn_head": ("#d8e8ff", "#5d7fbf"),
        "o_proj":    ("#d8e8ff", "#5d7fbf"),
        "mlp_in":    ("#fff5cc", "#b8963b"),
        "down_proj": ("#fff5cc", "#b8963b"),
        "output":    ("#ffd7b8", "#c47f42"),
        "mlp":       ("#fff5cc", "#b8963b"),
        "logits":    ("#eadcff", "#6b5fbf"),
    }
    return palette.get(typ, ("#eeeeee", "#888888"))


def _max_layer(nx_graph: nx.DiGraph) -> int:
    maxL = 0
    for n in nx_graph.nodes():
        try:
            maxL = max(maxL, int(nx_graph.nodes[n].get("layer", 0)))
        except Exception:
            pass
    return maxL


def _map_node_id_and_label(attrs: Dict) -> Tuple[str, str, str]:
    L   = int(attrs.get("layer", 0))
    typ = attrs.get("typ", "output")
    idx = int(attrs.get("idx", 0))

    if typ == "input":
        return "INPUT_GLOBAL", "input", "input"

    # 合并所有 MLP/残差输出类到一个 m{L}
    if typ in {"mlp_in", "down_proj", "output", "o_proj"}:
        return f"M{L}", f"m{L}", "mlp"

    if typ == "attn_head":
        return f"A{L}H{idx}", f"a{L}.h{idx}", "attn_head"

    return f"M{L}", f"m{L}", "mlp"


def _edge_style(weight: float, strong_cut: float) -> Tuple[str, float]:
    """根据权重正负与强弱设定颜色和粗细：正→蓝，负→红，弱→灰，极强→黑"""
    w = float(abs(weight))
    if w >= strong_cut:
        return ("#000000", 2.2)
    if weight > 0:
        return ("#3b6bd6", 1.3 + 1.2 * w)
    if weight < 0:
        return ("#d64b4b", 1.3 + 1.2 * w)
    return ("#888888", 1.0)


def _collect_input_targets_after_merge(declared_ids: set, edge_dict: Dict) -> List[str]:
    has_incoming = {v for (_, v) in edge_dict.keys()}
    targets = set(declared_ids) - has_incoming - {"INPUT_GLOBAL", "LOGITS"}
    return sorted(targets)


def _collect_logits_sources(nx_graph: nx.DiGraph) -> List[str]:
    """作为 logits 的入边来源：最大层上的节点 + 无出边节点（合并后名称）"""
    Lmax = _max_layer(nx_graph)
    ids = set()
    for n in nx_graph.nodes():
        attrs = nx_graph.nodes[n]
        L = int(attrs.get("layer", 0))
        if L == Lmax:
            nid, _, _ = _map_node_id_and_label(attrs)
            ids.add(nid)
    for n in nx_graph.nodes():
        if nx_graph.out_degree(n) == 0:
            nid, _, _ = _map_node_id_and_label(nx_graph.nodes[n])
            ids.add(nid)
    ids.discard("LOGITS")
    return sorted(ids)


def visualize_circuit_gv(nx_graph: nx.DiGraph, out_file: str) -> None:
    """Graphviz 渲染；不可用时回退为 NetworkX 简图（保持原行为）"""
    try:
        import graphviz
    except Exception as e:
        print("[viz] graphviz 不可用，回退到 networkx：", e)
        return visualize_circuit_fallback(nx_graph, out_file=out_file)

    nodes = list(nx_graph.nodes())
    ws = [abs(d.get("weight", 0.0)) for _, _, d in nx_graph.edges(data=True)]
    strong_cut = np.quantile(ws, 0.9) if len(ws) >= 4 else (max(ws) if ws else 1.0)

    dot = graphviz.Digraph("circuit", format="png")
    dot.attr(rankdir="LR", splines="spline", concentrate="false",
             nodesep="0.35", ranksep="1.0", bgcolor="white",
             fontname="Helvetica")

    declared: set = set()
    layer_groups: Dict[int, set] = {}
    node_colors: Dict[str, Tuple[str, str]] = {}  # 新增：记录 {node_id: (fill, border)}

    # 声明映射后的节点
    for n in nodes:
        attrs = nx_graph.nodes[n]
        node_id, label, color_typ = _map_node_id_and_label(attrs)
        if node_id in declared:
            continue
        fill, border = _node_style_for_typ(color_typ)
        dot.node(node_id, label=label, shape="box", style="rounded,filled",
                 color=border, penwidth="1.4", fillcolor=fill, fontsize="18")
        node_colors[node_id] = (fill, border)  # ← 记录颜色
        declared.add(node_id)
        if node_id != "INPUT_GLOBAL":
            L = int(attrs.get("layer", 0))
            layer_groups.setdefault(L, set()).add(node_id)

    # 各层 rank=same
    for L, ids in sorted(layer_groups.items()):
        with dot.subgraph(name=f"rank_{L}") as s:
            s.attr(rank="same")
            for nid in sorted(ids):
                s.node(nid)

    # INPUT（最左）
    input_id = "INPUT_GLOBAL"
    fill, border = _node_style_for_typ("input")
    dot.node(input_id, label="input", shape="box", style="rounded,filled",
             color=border, penwidth="1.4", fillcolor=fill, fontsize="18")
    node_colors[input_id] = (fill, border)  # 记录
    with dot.subgraph(name="rank_input") as s:
        s.attr(rank="min")
        s.node(input_id)

    # LOGITS（末层右侧）
    logits_id = "LOGITS"
    fill, border = _node_style_for_typ("logits")
    dot.node(logits_id, label="logits", shape="box", style="rounded,filled",
             color=border, penwidth="1.4", fillcolor=fill, fontsize="18")
    node_colors[logits_id] = (fill, border)
    if layer_groups:
        with dot.subgraph(name=f"rank_{max(layer_groups.keys()) + 1}") as s:
            s.attr(rank="same")
            s.node(logits_id)
    
    def _edge_color_from_node(node_id: str) -> str:
        # 优先取“边框色”，没有就取“填充色”，再没有就灰
        fill_col, border_col = node_colors.get(node_id, (None, None))
        if border_col and isinstance(border_col, str) and border_col.startswith("#"):
            return border_col
        if fill_col and isinstance(fill_col, str) and fill_col.startswith("#"):
            return fill_col
        return "#9aa0a6"

    # 聚合同一对节点的多条边
    edge_dict: Dict[Tuple[str, str], List[float]] = {}
    for u, v, d in nx_graph.edges(data=True):
        u_id, _, _ = _map_node_id_and_label(nx_graph.nodes[u])
        v_id, _, _ = _map_node_id_and_label(nx_graph.nodes[v])
        if u_id == v_id:
            continue
        w = float(d.get("weight", 0.0))

        edge_dict.setdefault((u_id, v_id), []).append(w)

    # —— 新增：按“合并后源节点”累计所有外发边的权重，用于 LOGITS 边加权 ——
    node_out_weights: Dict[str, List[float]] = {}
    for (u_id, v_id), ws_ in edge_dict.items():
        node_out_weights.setdefault(u_id, []).extend(ws_)

    # 画边（对同一对端点取平均权重）
    for (u_id, v_id), ws_ in edge_dict.items():
        w = float(np.mean(ws_))
        # 线宽仍按强度；颜色继承出发节点
        _, penw = _edge_style(w, strong_cut)
        edge_col = _edge_color_from_node(u_id)
        attrs = {"color": edge_col, "penwidth": f"{penw:.2f}", "arrowsize": "0.7"}
        dot.edge(u_id, v_id, **attrs)

    # INPUT → “在合并后无任何入边”的节点（颜色继承 INPUT）
    declared_ids_now = set(declared) | {"LOGITS"}
    input_targets = _collect_input_targets_after_merge(declared_ids_now, edge_dict)
    _, inp_border = node_colors.get(input_id, ("#d7f0d0", "#6ca966"))
    for tid in input_targets:
        if tid == "INPUT_GLOBAL":
            continue
        dot.edge(input_id, tid, color=inp_border, penwidth="1.1", arrowsize="0.8")

    for sid in _collect_logits_sources(nx_graph):
        if sid == logits_id:
            continue
        ws_ = node_out_weights.get(sid, [])
        edge_col = _edge_color_from_node(sid)  # ← 继承源节点颜色
        if ws_:
            w_signed = float(np.mean(ws_))
            w_abs    = float(np.mean(np.abs(ws_)))
            _, penw = _edge_style(w_signed, strong_cut)  # 只用线宽刻度
            penw = max(penw, 0.9 + 1.1 * min(w_abs / (strong_cut + 1e-8), 1.0))
            attrs = {"color": edge_col, "penwidth": f"{penw:.2f}", "arrowsize": "0.8"}
        else:
            attrs = {"color": edge_col, "penwidth": "0.9", "arrowsize": "0.8"}
        dot.edge(sid, logits_id, **attrs)

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dot.render(filename=out_path.with_suffix("").as_posix(), cleanup=True)
    dot.render(filename=out_path.with_suffix(".pdf").as_posix(), format="pdf", cleanup=True)
    print(f"[viz] saved: {out_file}")


def visualize_circuit_fallback(nx_graph: nx.DiGraph, out_file: str) -> None:
    """在不装 graphviz 时的简易回退（networkx draw）。"""
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(nx_graph, seed=0)
    plt.figure(figsize=(10, 6))
    nx.draw(nx_graph, pos, with_labels=False, node_size=300, width=0.8, arrows=True)
    plt.tight_layout()
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=FIG_DPI)
    plt.close()
    print(f"[viz-fallback] saved: {out_file}")


def visualize_top_edges_as_circuit(
    graph: "EAPGraph",
    out_file: str,
    n: int = TOP_N_EDGES,
    abs_scores: bool = True,
    include_attn: bool = True,
    include_mlp: bool = True,
) -> None:
    """从 EAPGraph → NX → Graphviz PNG（参考图风格）"""
    # 调试预览解析效果
    for u, v, s in graph.top_edges(n=min(8, n), abs_scores=True):
        uL, uT, uI = _parse_eap_node_relaxed(u)
        vL, vT, vI = _parse_eap_node_relaxed(v)
        print(f"[parsed] U=({uL},{uT},{uI})  V=({vL},{vT},{vI})  |S|={abs(s):.3f}   rawU={u}  rawV={v}")

    G = build_nx_from_top_edges(
        graph, n=n, abs_scores=abs_scores,
        include_attn=include_attn, include_mlp=include_mlp,
    )
    G = _prune_keep_bidir_with_context(G)
    visualize_circuit_gv(G, out_file=out_file)


# ───────────────────────────── CLI & Main ──────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="TransformerLens 兼容模型名（如：gpt2-small 或 TL-ported 名称）")
    p.add_argument("--data_file", default=DEFAULT_DATA,
                   help="JSON：logic_groups[*].prompts[*].{clean,corrupt,...}")
    p.add_argument("--quants", nargs="+", type=float, default=[0.5],
                   help="分位阈值（取 >= q 对应的分数，进而“保留 (1-q) 比例”）")
    p.add_argument("--subset_k", type=int, default=10,
                   help="每个 split 中 partA/partB 各取 k 条样本")
    p.add_argument("--splits", type=int, default=1, help="随机切分次数")
    p.add_argument("--seed", type=int, default=0, help="随机种子基数")
    p.add_argument("--batch_size", type=int, default=1, help="EAP 前反向 batch 大小")
    p.add_argument("--out_root", default=DEFAULT_OUT, help="结果根目录")
    p.add_argument("--resume", type=Path, help="可选：state_dict 路径（尽量灌入 TL 模型）")
    p.add_argument("--strict_resume", action="store_true", help="严格 key 对齐加载 state_dict")
    p.add_argument("--ans_true", default=" True", help="avg_diff 正类词（默认 ' True'）")
    p.add_argument("--ans_false", default=" False", help="avg_diff 负类词（默认 ' False'）")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 种子：注意 split 内部仍会基于 seed 偏移
    set_seed_all(args.seed)

    # 1) 加载 TL 模型
    print(f"[load] HookedTransformer.from_pretrained({args.model}) on {device}")
    model = HookedTransformer.from_pretrained(args.model, device=device)

    # 有些配置项可能不存在，包容式设置
    try:
        model.cfg.parallel_attn_mlp = False
    except Exception:
        pass
    add_identity_mid_hooks(model)

    # 2) resume（若有）
    run_tag = Path(args.resume).stem if args.resume else "origin"
    if args.resume is not None:
        try_resume_state_dict(model, args.resume, args.strict_resume)
    print(f"[info] Using run tag: {run_tag}")

    # 3) 目录
    out_root = Path(args.out_root).resolve()
    dataset_tag = Path(args.data_file).stem
    model_dir = out_root / dataset_tag / model_tag(args.model)
    for q in args.quants:
        (model_dir / quant_tag(q) / run_tag / "figures").mkdir(parents=True, exist_ok=True)

    # 4) 读取数据
    with open(args.data_file, "r", encoding="utf-8") as f:
        logic_groups = json.load(f)

    # 5) 遍历逻辑组
    for idx, logic in enumerate(logic_groups):
        examples = logic.get("prompts", [])
        if not examples:
            continue

        logic_name = f"logic_{idx:03d}"
        if idx < 8:
            continue
        print(f"\n▶ {logic_name} — {len(examples)} prompts")

        # 6) 抽样切分
        k = min(args.subset_k, max(len(examples) // 2, 1))
        if dataset_tag == "level_3":
            k = min(k, 4)

        for split in range(1, args.splits + 1):
            seed_i = args.seed + split - 1
            rng = random.Random(seed_i)
            shuffled = examples[:]
            # rng.shuffle(shuffled)
            partA, partB = shuffled[:k], shuffled[k: 2 * k]

            # 6.1) A 子集
            clean_A, corrupt_A = extract_clean_corrupt_pairs(partA)
            graph_A = eap_scores_for_subset(
                model, clean_A, corrupt_A, args.ans_true, args.ans_false, args.batch_size
            )
            for q in args.quants:
                qtag = quant_tag(q)
                subdir = model_dir / qtag / run_tag
                json_path = subdir / f"{logic_name}_split{seed_i}_partA.json"
                save_json(threshold_and_sparse(graph_A, q), json_path)

                viz_dir = subdir / "figures"
                viz_dir.mkdir(parents=True, exist_ok=True)
                visualize_top_edges_as_circuit(
                    graph_A,
                    out_file=str(viz_dir / f"{logic_name}_split{seed_i}_partA_circuit_top100.png"),
                    n=TOP_N_EDGES,
                    abs_scores=True,
                    include_attn=True,
                    include_mlp=True,
                )
                heat_path = viz_dir / f"{logic_name}_split{seed_i}_partA.png"
                save_heatmap(graph_A.eap_scores, heat_path, title=f"{logic_name} partA (q={q:.3f})")
                print(f"   ✓ partA  q={q:.3f} → {json_path.relative_to(model_dir)}")

            # 6.2) B 子集（若有）
            if partB:
                clean_B, corrupt_B = extract_clean_corrupt_pairs(partB)
                graph_B = eap_scores_for_subset(
                    model, clean_B, corrupt_B, args.ans_true, args.ans_false, args.batch_size
                )
                for q in args.quants:
                    qtag = quant_tag(q)
                    subdir = model_dir / qtag / run_tag
                    json_path = subdir / f"{logic_name}_split{seed_i}_partB.json"
                    save_json(threshold_and_sparse(graph_B, q), json_path)

                    viz_dir = subdir / "figures"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    visualize_top_edges_as_circuit(
                        graph_B,
                        out_file=str(viz_dir / f"{logic_name}_split{seed_i}_partB_circuit_top100.png"),
                        n=TOP_N_EDGES,
                        abs_scores=True,
                        include_attn=True,
                        include_mlp=True,
                    )
                    heat_path = viz_dir / f"{logic_name}_split{seed_i}_partB.png"
                    save_heatmap(graph_B.eap_scores, heat_path, title=f"{logic_name} partB (q={q:.3f})")
                    print(f"   ✓ partB  q={q:.3f} → {json_path.relative_to(model_dir)}")

    print("\n✅ Finished for all TOP_QUANTS.")


if __name__ == "__main__":
    main()
