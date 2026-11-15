#!/usr/bin/env bash
# Reproduce one reference experiment: OQMD formation energy, PCG features and ranking, SVR,
# all eligible held-out elements, five repetitions, filtered tests. Then plot the result,
# next to the published reference result when it has been downloaded.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:?Usage: bash run_oqmd_svr.sh DATA_DIR OUT_DIR [cpu|gpu]}"
OUT_DIR="${2:?Usage: bash run_oqmd_svr.sh DATA_DIR OUT_DIR [cpu|gpu]}"
DEVICE="${3:-cpu}"
REFERENCE="$DATA_DIR/reference_results/ablation_filtered/pcg/ablation_OQMD_e_form_all_eSEN_layer4_SVR_base10000_testtop6_81ecfc423286.csv"

python "$SCRIPT_DIR/../scripts/run_ablation.py" \
  --data-dir "$DATA_DIR" --out-dir "$OUT_DIR" \
  --db OQMD --task e_form --representation pcg --ranking pcg \
  --method SVR --device "$DEVICE" --train-size 10000 --repeats 5

INPUTS=("$OUT_DIR")
if [[ -f "$REFERENCE" ]]; then
  INPUTS+=("$REFERENCE")
fi
python "$SCRIPT_DIR/../scripts/plot_ablation.py" --inputs "${INPUTS[@]}" --out-dir "$OUT_DIR/figures"
