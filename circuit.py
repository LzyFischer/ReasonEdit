import json
from pathlib import Path

import torch as t
from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.visualize import draw_seq_graph

# ─── Configuration ─────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
DATA_DIR   = Path("data/corrupt")
OUT_JSON   = Path("output/attr_scores")
OUT_FIGS   = OUT_JSON / "figures"
BATCH_SIZE = 2
TOP_QUANT  = 0.999  # keep top 0.5%

OUT_JSON.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

# ─── Load & patch model ─────────────────────────────────────────────────────
device = t.device("cuda" if t.cuda.is_available() else "cpu")
base_model = load_tl_model(MODEL_NAME, device)
model = patchable_model(
    base_model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=True,
    device=device,
)

# ─── Iterate over all logic_*.json files ───────────────────────────────────
for json_path in sorted(DATA_DIR.glob("logic_1_all.json")):
    name = json_path.stem
    print(f"\n▶ Processing {name}")

    # 1) Load prompts as dataset
    len_prompt = len(json.load(open(json_path, "r"))["prompts"])
    train_loader, test_loader = load_datasets_from_json(
        model=model,
        path=json_path,
        device=device,
        prepend_bos=True,
        batch_size=len_prompt // 3 if len_prompt > 3 else 1,  # ensure at least batch size of 1
        train_test_size=(len_prompt, 1),
    )

    # 2) Compute attribution scores
    attribution_scores: PruneScores = mask_gradient_prune_scores(
        model=model,
        dataloader=train_loader,
        official_edges=None,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    )

    # 3) Compute the quantile threshold for top-0.5%
    all_pos = t.cat([
        t.tensor(v).flatten()[t.tensor(v).flatten() > 0.0]
        for v in attribution_scores.values()
    ], dim=0)
    top_thresh = all_pos.quantile(TOP_QUANT).item() if all_pos.numel() > 0 else 0.0
    print(f"   → Keeping edges ≥ {top_thresh:.4f} (top 0.5%)")

    # 4) Zero-out everything below threshold
    POS_TOP: PruneScores = {
        k: (t.tensor(v) * (t.tensor(v) >= top_thresh).to(t.float32))
        for k, v in attribution_scores.items()
    }

    # 5) Draw & save Sankey diagram
    fig = draw_seq_graph(
        model,
        POS_TOP,
        score_threshold=top_thresh,
        layer_spacing=True,
        orientation="v",
        # display_ipython=False
    )

    # 6) Save the masked scores JSON
    out_json = OUT_JSON / f"{name}.json"
    with open(out_json, "w") as f:
        json.dump({k: v.tolist() for k, v in POS_TOP.items()}, f, indent=2)
    print(f"   ✓ Masked scores saved to {out_json}")

print("\n✅ All files processed.")