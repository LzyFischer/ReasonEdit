#!/usr/bin/env bash
set -euo pipefail
# cd ../..

seeds=(5 10 15)
lrs=(1e-4)
dists=(pot edit jaccard)

for s in "${seeds[@]}"; do
  python -m src.preliminary.circuit.circuit_aio --seed "$s"
done

for s in "${seeds[@]}"; do
  for d in "${dists[@]}"; do
    python -m "src.preliminary.distance.${d}_distance" --seed "$s"
  done
done

for lr in "${lrs[@]}"; do
  python -m src.preliminary.edit.perlogic_delta --lr "$lr"
done

for s in "${seeds[@]}"; do
  for d in "${dists[@]}"; do
    for lr in "${lrs[@]}"; do
      python -m tools.plot_preliminary --distance "$d" --seed "$s" --lr "$lr"
    done
  done
done
