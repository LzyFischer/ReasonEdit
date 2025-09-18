from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, List, Optional

NON_CATEGORY_KEYS = {"question", "answer"}

def filter_level2_record(
    rec: dict,
    keep_categories: Optional[List[str]] = None,
    m_categories: Optional[int] = None,
    keep_map: Optional[Dict[str, List[str]]] = None,
    k_per_category: Optional[int] = None,
    rng: random.Random = random.Random(0),
) -> dict:
    """对单条样本进行过滤：类目级 + 子主题级"""
    out = {}
    for k in rec:
        if k in NON_CATEGORY_KEYS:
            out[k] = rec[k]

    all_cats = [k for k, v in rec.items() if k not in NON_CATEGORY_KEYS and isinstance(v, dict)]

    # 类目过滤
    if keep_categories is not None:
        cats = [c for c in all_cats if c in keep_categories]
    else:
        cats = list(all_cats)
    if m_categories is not None and len(cats) > m_categories:
        rng.shuffle(cats)
        cats = cats[:m_categories]

    # 子主题过滤
    for cat in cats:
        subtopics = rec[cat]
        if keep_map and cat in keep_map:
            allow = set(keep_map[cat])
            filtered = {name: payload for name, payload in subtopics.items() if name in allow}
        else:
            filtered = dict(subtopics)
        if k_per_category is not None and len(filtered) > k_per_category:
            names = list(filtered.keys())
            rng.shuffle(names)
            keep_names = set(names[:k_per_category])
            filtered = {name: filtered[name] for name in keep_names}
        if filtered:
            out[cat] = filtered

    return out


def build_subset(
    in_path: str | Path,
    out_path: str | Path,
    n_records: Optional[int] = None,            # 样本级随机抽取
    keep_categories: Optional[List[str]] = None,
    m_categories: Optional[int] = None,
    keep_map: Optional[Dict[str, List[str]]] = None,
    k_per_category: Optional[int] = None,
    seed: int = 0,
):
    data = json.loads(Path(in_path).read_text())
    rng = random.Random(seed)

    # -------- 样本级抽取 --------
    if n_records is not None and len(data) > n_records:
        # rng.shuffle(data)
        data = data[:n_records]

    # -------- 类目 & 子主题级抽取 --------
    subset = [
        filter_level2_record(
            rec,
            keep_categories=keep_categories,
            m_categories=m_categories,
            keep_map=keep_map,
            k_per_category=k_per_category,
            rng=rng,
        )
        for rec in data
    ]
    Path(out_path).write_text(json.dumps(subset, ensure_ascii=False, indent=2))


# ---------------- 使用示例 ----------------
# A) 样本级抽取 100 条
# build_subset("level_2.json", "subset_records.json", n_records=100, seed=42)

# B) 样本级抽 50 条 + 每条随机抽 3 个一级类目
build_subset("data/logic/level_3.json", "data/logic/level_3_1.json",
             n_records=10, seed=123)

# C) 同时控制三个层次：
#    1) 总样本数 80 条
#    2) 每条样本只保留 culture and arts, mathematics and logic
#    3) 每个类目里随机抽取 2 个子主题
# build_subset("level_2.json", "subset_all_levels.json",
#              n_records=80,
#              keep_categories=["culture and arts", "mathematics and logic"],
#              k_per_category=2, seed=7)

