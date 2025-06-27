# Preliminary
## Trade-off
1. Run `bash scripts/preliminary.sh` to obtain locality/generality acc under different learning rate.
2. Plot in `src/preliminary/notebooks/plot_preliminary.ipynb`.

## Fack-checking
1. Run `bash scripts/preliminary_fc.sh` to obtain performance of unfine-tuned fact-checking and reasoning acc.
2. Plot in `src/preliminary/notebooks/plot_preliminary.ipynb`.

## Motivation evidence
1. Run `bash scripts/circuit.sh` to obtain the circuit of all reasoning paths.
2. Run `python -m src.preliminary.edit.perlogic_delta` to obtain locality/generality of each reasoning path.
3. Plot in `src/preliminary/notebooks/plot_preliminary.ipynb`.
