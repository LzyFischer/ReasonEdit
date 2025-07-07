# Preliminary Workflow
##### Note: all scripts (sh) you need to modify to change the model/learning rate
## 1 · Locality-Generality Trade-off  
- **Run** `bash scripts/preliminary.sh` to compute accuracy across learning-rates.  
- **Visualise** results in `src/preliminary/notebooks/plot_preliminary.ipynb`.

## 2 · Fact-Checking Baseline  
- **Execute** `bash scripts/preliminary_fc.sh` to gather raw fact-checking and reasoning scores (no fine-tuning).  
- **Plot** with the same notebook: `plot_preliminary.ipynb`.

## 3 · Motivation & Evidence  
1. `python -m src.preliminary.data_gen.generate_reverse` - generate corrupted prompts.
2. `bash scripts/circuit.sh` — extract circuits for every reasoning path.  
3. `bash scripts/distance.sh` — compute inter-circuit distances.  
4. `python -m src.preliminary.edit.perlogic_delta` — evaluate locality & generality per path.  
5. **Chart** outcomes in `plot_preliminary.ipynb`.
5. **Or you can directly run `python tools/run_preliminary_pipeline.py` all in one go.**