from pathlib import Path

# ── project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # …/ReasonEdit-main

# ── primary sub-directories ───────────────────────────────────────────────
DATA_DIR     = PROJECT_ROOT / "data"                    # …/data
RESULTS_DIR  = PROJECT_ROOT / "results"                 # …/results

# Anything previously under "output/" now lives inside results/output/
OUTPUTS_DIR  = RESULTS_DIR / "output"                   # …/results/output
ATTR_SCORES_DIR = OUTPUTS_DIR / "attr_scores"           # …/results/output/attr_scores