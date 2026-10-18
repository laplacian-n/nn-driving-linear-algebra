#!/bin/bash
# Show model with 17 rays + PCA 8: BC base (2 seeds) -> pick best on T -> showcase BC->GA (3 seeds)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in 0 1; do
  $PY -m racing.bc --name bc_t75_long/P8_XL_2x64 --rays 17 --reduction pca --k 8 --depth 2 --width 64 --seed $s \
      --aggregate all --teacher-grip 0.75 --teacher-brake 0.62 --steps 3000 --patience 8 \
      > experiments/results/bc_pca8_xl_s$s.log 2>&1 &
done
wait
BEST=$($PY - <<'PYEOF'
import json
import numpy as np
f = lambda s: np.mean(json.load(open(f"experiments/results/bc_t75_long/P8_XL_2x64/seed{s}/final_test.json"))["finished"])
best = max((0, 1), key=f)
print(best)
PYEOF
)
echo "base seed $BEST"
for s in 0 1 2; do
  $PY -m racing.showcase --seed $s --deadline 1792334700 --base experiments/results/bc_t75_long/P8_XL_2x64/seed$BEST/station_2.00.npz \
      --out experiments/results/showcase_pca8 > experiments/results/showcase_pca8_s$s.log 2>&1 &
done
wait
echo done
