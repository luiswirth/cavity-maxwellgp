#!/bin/bash
set -euo pipefail
if [[ $# -eq 0 ]]; then set -- ellipse sphere; fi
mkdir -p out/logs
uv sync --quiet
for geom in "$@"; do
  rm -rf out/noise-grid/$geom
  sbatch --array=1-39%4 euler/run.sbatch "$geom" noise-grid euler/noise_grid.txt
done
