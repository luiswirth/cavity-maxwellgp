#!/bin/bash
# Run the EP-GP convergence grid (or field slice / noise sweep) locally, without SLURM.
# Serial port of euler/run.sbatch for reproduction on any machine.
#
#   ./run_local.sh grid  [geom...]   # 2D (N_s, N_b) convergence grid
#   ./run_local.sh field [geom...]   # field slice at the highest-fidelity grid config
#   ./run_local.sh noise [geom...]   # fixed-resolution sweep over assumed noise
#   ./run_local.sh noise-grid [geom...]  # N_s sweep at each of several assumed noise levels
#
# geom defaults to: ellipse sphere. Requires uv (https://docs.astral.sh/uv/).
set -euo pipefail

MODE="${1:-grid}"; shift || true
case "$MODE" in
  grid)  TASKFILE=euler/grid.txt;;
  field) TASKFILE=euler/field.txt;;
  noise) TASKFILE=euler/noise.txt;;
  noise-grid) TASKFILE=euler/noise_grid.txt;;
  *) echo "usage: $0 grid|field|noise|noise-grid [geom...]" >&2; exit 1;;
esac
[[ $# -eq 0 ]] && set -- ellipse sphere

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
uv sync --quiet

for geom in "$@"; do
  base="$ROOT/out/$MODE/$geom"
  rm -rf "$base"; mkdir -p "$base"
  while read -r NS NB LN; do
    R="$base"; NOISE_ARG=()
    if [[ -n "${LN:-}" ]]; then
      R="$base/ln${LN}"; NOISE_ARG=(--log-noise "$LN")
      mkdir -p "$R"
    fi
    SECONDS=0
    log=$(uv run --project "$ROOT" epgp-operator operator \
      --config "$ROOT/res/config_${geom}.txt" --n-spectral "$NS" --n-boundary "$NB" \
      --outdir "$R" ${NOISE_ARG[@]+"${NOISE_ARG[@]}"} 2>&1)
    echo "$log"
    dofs=$(echo "$log" | grep -oE 'dofs=[0-9]+' | grep -oE '[0-9]+')
    cond=$(echo "$log" | grep -oE 'cond=[0-9a-z.E+-]+' | grep -oE '[0-9a-z.E+-]+$')
    log_noise=$(echo "$log" | grep -oE 'log_noise=[0-9a-z.E+-]+' | grep -oE '[0-9a-z.E+-]+$')
    nlml=$(echo "$log" | grep -oE 'nlml=[0-9a-z.E+-]+' | grep -oE '[0-9a-z.E+-]+$')
    # manifest columns: n_spectral,n_boundary,dofs,secs,mem_kb,cond  (mem_kb empty locally)
    echo "${NS},${NB},${dofs},${SECONDS},,${cond}" >> "$R/manifest.csv"
    echo "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown),$(hostname),${NS},${NB},$(date -Is),${log_noise},${nlml}" >> "$R/provenance.csv"
    echo "done ns${NS}nb${NB}: dofs=$dofs cond=$cond secs=${SECONDS}"
    if [[ "$MODE" == field ]]; then
      uv run --project "$ROOT" epgp-operator field \
        --config "$ROOT/res/config_${geom}.txt" --n-spectral "$NS" --n-boundary "$NB" \
        --source 0 0 1 --pol 1 0 0 --out "$R/field.npz"
    fi
  done < <(grep -vE '^\s*#|^\s*$' "$ROOT/$TASKFILE")
done
