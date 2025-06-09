import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import draw_seq_graph

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = load_tl_model("gpt2", device)

path = repo_path_to_abs_path("data/ioi/ioi_vanilla_template_prompts.json")
train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=16,
    train_test_size=(128, 128),
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=True,
    device=device,
)
