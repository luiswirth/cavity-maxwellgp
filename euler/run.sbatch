#!/bin/bash
# === SLURM site config (ETH Euler) ===========================================
# Edit --account and --constraint below for a different cluster; drop them to
# run locally via ../run_local.sh.
#SBATCH --job-name=epgp
#SBATCH --account=ls_math
#SBATCH --output=out/logs/slurm-%A_%a.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=2G
#SBATCH --exclusive
#SBATCH --constraint=EPYC_7742
# =============================================================================
#
#   euler/submit_grid.sh  [geom]  # convergence grid
#   euler/submit_field.sh [geom]  # field slice at the highest-fidelity grid config
#   euler/submit_noise.sh [geom]  # fixed-resolution sweep over assumed noise
#   euler/submit_noise_grid.sh [geom]  # N_s sweep at each of several assumed noise levels
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
GEOM=$1; MODE=$2; TASKFILE=$3
ROOT=$(pwd)
# taskfile columns: n_spectral n_boundary [log_noise]; a log_noise puts the run in ln<log_noise>/
read -r NS NB LN < <(grep -vE '^\s*#|^\s*$' "$ROOT/$TASKFILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")
R=$ROOT/out/$MODE/$GEOM; NOISE_ARG=""
if [[ -n "${LN:-}" ]]; then R=$R/ln${LN}; NOISE_ARG="--log-noise ${LN}"; fi
mkdir -p "$R" "$ROOT/out/logs"
work=$R/work/ns${NS}_nb${NB}; rm -rf "$work"; mkdir -p "$work"; cd "$work"
srun --cpu-bind=cores /usr/bin/time -f "elapsed=%e mem=%M" -o time.txt \
  uv run --project "$ROOT" epgp-operator operator \
  --config "$ROOT/res/config_${GEOM}.txt" --n-spectral "$NS" --n-boundary "$NB" --outdir "$R" ${NOISE_ARG} \
  > run.log 2>&1
log=$(cat run.log); echo "$log"
dofs=$(echo "$log" | grep -oE 'dofs=[0-9]+' | grep -oE '[0-9]+')
cond=$(echo "$log" | grep -oE 'cond=[0-9a-z.E+-]+' | grep -oE '[0-9a-z.E+-]+$')
log_noise=$(echo "$log" | grep -oE 'log_noise=[0-9a-z.E+-]+' | grep -oE '[0-9a-z.E+-]+$')
nlml=$(echo "$log" | grep -oE 'nlml=[0-9a-z.E+-]+' | grep -oE '[0-9a-z.E+-]+$')
secs=$(grep -oE 'elapsed=[0-9.]+' time.txt | grep -oE '[0-9.]+' | cut -d. -f1 || true)
mem_kb=$(grep -oE 'mem=[0-9]+' time.txt | grep -oE '[0-9]+' || true)
# manifest columns: n_spectral,n_boundary,dofs,secs,mem_kb,cond
echo "${NS},${NB},${dofs},${secs},${mem_kb},${cond}" >> "$R/manifest.csv"
# provenance columns: git,host,n_spectral,n_boundary,date,log_noise,nlml
echo "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown),$(hostname),${NS},${NB},$(date -Is),${log_noise},${nlml}" >> "$R/provenance.csv"
echo "done ns${NS}nb${NB}: dofs=$dofs cond=$cond secs=$secs mem_kb=$mem_kb"
if [[ "$MODE" == "field" ]]; then
  srun uv run --project "$ROOT" epgp-operator field \
    --config "$ROOT/res/config_${GEOM}.txt" --n-spectral "$NS" --n-boundary "$NB" \
    --source 0 0 1 --pol 1 0 0 \
    --out "$R/field.npz" >> run.log 2>&1
  echo "field slice written to $R/field.npz"
fi
