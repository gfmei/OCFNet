#!/bin/bash
#SBATCH --job-name=eval_sweep
# 1 h, not 4 h: backfill only fits a job into a gap at least as long as its request, and
# measured over 100+ jobs a 4 h request waits a median of 48 min in the queue while a 40 min
# request waits 0. The sweep takes 10-21 min on 32 cores; RANSAC parallelises over pairs so
# 16 cores roughly doubles that, which is why this is 1 h rather than 40 min.
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=64G
#SBATCH --partition=boost_usr_prod
# No --account here: pass `sbatch --account=<your account> ...`, or set a default with
# `sacctmgr modify user $USER set DefaultAccount=<account>`.
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source "${CONDA_ROOT:-$(conda info --base)}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-reg3d}"
# The evaluation is RANSAC-bound and parallelises over pairs, one thread each
# (scripts/evaluate_ocfnet.py); throttling OMP here would cap that instead.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Repo root. Under sbatch the script runs from a spool copy, so BASH_SOURCE points at the
# spool directory, not the repo -- SLURM_SUBMIT_DIR is the directory sbatch was called from.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIG="${1:-${CONFIG:-configs/test/indoor.yaml}}"
test -f "$CONFIG" || { echo "no such config: $CONFIG" >&2; exit 2; }
BENCH="${BENCH:-3DMatch}"
SAMPLE_LIST="${SAMPLE_LIST:-5000 2500 1000 500 250}"
EXP=$(grep -E '^\s+exp_dir:' "$CONFIG" | awk '{print $2}')
TMP="${TMPDIR:-/tmp}/ocfnet"
mkdir -p "$TMP"

# The per-pair features do not depend on the number of sampled correspondences, so the
# expensive inference pass runs once and every sample count reuses the same dump.
RUN_CONFIG="$TMP/$(basename "${CONFIG%.yaml}")_sweep_$BENCH.yaml"
# batch_size must be 1: write_est_trajectory indexes the estimates per *pair*, so a
# batched loader yields ceil(pairs/B) transforms and the write walks off the end.
sed -e "s|^  benchmark: .*|  benchmark: $BENCH|" \
    -e "s|^\(\s*\)batch_size: .*|\1batch_size: 1|" "$CONFIG" > "$RUN_CONFIG"
echo "########## $BENCH, dumping features once for samples: $SAMPLE_LIST"
python -u main.py "$RUN_CONFIG"

# Results land in est_traj/<benchmark>/<samples>/result. Clear this benchmark's tree first:
# a previous evaluation of an older checkpoint leaves files there, and a sweep that does not
# reach every sample count would otherwise mix stale numbers in with the new ones.
rm -rf "snapshot/$EXP/est_traj/$BENCH"

MODEL=$(grep -E '^\s+model:' "$CONFIG" | awk '{print $2}')
if [ "$MODEL" = "OCFNet" ]; then EVAL=scripts/evaluate_ocfnet.py; else EVAL=scripts/evaluate_predator.py; fi
for N in $SAMPLE_LIST; do
  echo "########## $BENCH @ $N samples"
  python -u "$EVAL" --source_path "snapshot/$EXP/$BENCH" --n_points "$N" \
      --benchmark "$BENCH" --exp_dir "snapshot/$EXP/est_traj"
done
rm -rf "snapshot/$EXP/$BENCH"     # the dump is large and no longer needed
