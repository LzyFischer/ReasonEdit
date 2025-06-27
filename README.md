# Preliminary Workflow
##### Note: all scripts (sh) you need to modify to change the model/learning rate
## 1 · Locality-Generality Trade-off  
- **Run** `bash scripts/preliminary.sh` to compute accuracy across learning-rates.  
- **Visualise** results in `src/preliminary/notebooks/plot_preliminary.ipynb`.

## 2 · Fact-Checking Baseline  
- **Execute** `bash scripts/preliminary_fc.sh` to gather raw fact-checking and reasoning scores (no fine-tuning).  
- **Plot** with the same notebook: `plot_preliminary.ipynb`.

## 3 · Motivation & Evidence  
1. `bash scripts/circuit.sh` — extract circuits for every reasoning path.  
2. `bash scripts/distance.sh` — compute inter-circuit distances.  
3. `python -m src.preliminary.edit.perlogic_delta` — evaluate locality & generality per path.  
4. **Chart** outcomes in `plot_preliminary.ipynb`.