#!/bin/bash
#SBATCH --job-name=ocfnet_indoor
# 24 h, the partition maximum. Re-measured with sbatch --test-only under a loaded queue:
# 4 h, 8 h, 12 h and 24 h all backfill within a minute of each other, and 24 h actually
# started earliest, so a longer slot costs nothing in queue time. It costs a third as many
# handoffs, and a handoff is not free -- a refused successor submission ended two runs after
# 7h50m each. boost_qos_lprod would allow 4 days but schedules far worse.
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=128G
#SBATCH --account=IscrC_ERAR --partition=boost_usr_prod
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@600
set -euo pipefail
source /leonardo_scratch/fast/AIFPT_agrifood/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-reg3d}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /leonardo_scratch/fast/AIFPT_agrifood/code/OCFNet

# Single GPU: the model is small and the data loader is the bottleneck, so the extra CPUs
# feed it instead. Override the config with CONFIG=configs/train/indoor.yaml for the
# upstream setting (batch_size 1, iter_size 4).
# config from the first argument, the CONFIG env var, or the batch-4 default.
# A wrong path must fail loudly here rather than silently training the default.
TRAIN_CONFIG="${1:-${CONFIG:-configs/train/predator_indoor.yaml}}"
test -f "$TRAIN_CONFIG" || { echo "no such config: $TRAIN_CONFIG" >&2; exit 2; }
EXP=$(grep -oP '(?<=exp_dir: ).*' "$TRAIN_CONFIG")
LAST="snapshot/$EXP/checkpoints/model_last.pth"
MAX_EPOCH=$(grep -oP '(?<=max_epoch: )[0-9]+' "$TRAIN_CONFIG")
# name the job after what it trains, so the queue cannot show one model as another
[ -n "${SLURM_JOB_ID:-}" ] && scontrol update JobId="$SLURM_JOB_ID" JobName="$EXP" 2>/dev/null
echo "training with $TRAIN_CONFIG -> snapshot/$EXP (to epoch $MAX_EPOCH)"

# 150 epochs take far longer than one allocation. SLURM raises USR1 ten minutes before the
# time limit (--signal above); only then is a successor queued, so a run that dies of a code
# error or OOM stops dead instead of spawning a chain that never trains anything. The
# trainer resumes from the per-epoch checkpoint.
if [ -f "$LAST" ]; then
  DONE=$(python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$LAST")
  if [ "$DONE" -ge "$MAX_EPOCH" ]; then
    echo "already at epoch $DONE of $MAX_EPOCH -- nothing to do"; exit 0
  fi
  echo "resuming from epoch $DONE"
fi

on_time_limit() {
  echo "time limit approaching, queueing a successor to resume this run"
  if [ "${CHAIN:-1}" = 1 ]; then
    # A failed submission here silently ends the run: two 8 h runs died this way when the
    # account hit its processor limit. Report it loudly instead.
    NEXT=$(sbatch --parsable --job-name="$SLURM_JOB_NAME" \
           scripts/slurm_train_indoor.sh "$TRAIN_CONFIG" 2>&1) \
      && echo "successor queued as $NEXT" \
      || echo "WARNING: could not queue a successor: $NEXT   resubmit by hand" >&2
  fi
  kill "$PY" 2>/dev/null
  exit 0
}
trap on_time_limit USR1

python -u main.py "$TRAIN_CONFIG" &
PY=$!
wait "$PY"
