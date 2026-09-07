#!/bin/bash
#SBATCH --job-name=ocfnet_indoor
# 2d08h under boost_qos_lprod (its ceiling is 4 days). A longer slot costs little queue
# time -- 4 h, 8 h, 12 h and 24 h all backfilled within a minute of each other under a loaded
# queue -- and it costs far fewer handoffs, which are not free: a refused successor
# submission ended two runs after 7h50m each. Anything over 24 h needs the qos line as well
# as the time line, or the job is rejected outright.
#SBATCH --time=2-08:05:05
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=128G
# Account comes from the ACCOUNT env var so a run can be moved off an account that
# is over its monthly allowance -- fair-share priority is driven by recent usage,
# so an over-quota account backfills last. `saldo -b` shows the balances.
# Do not use EUHPC_D30_012. Expired or exhausted: FBKLM_prj1, FBKLM_prj2,
# IscrC_3DLLM, IscrC_4grasp. Usable: AIFPT_agrifood, IscrC_TeVLA, IscrC_ERAR.
#SBATCH --account=AIFPT_agrifood --partition=boost_usr_prod
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
    # account hit its processor limit, and two more when sbatch produced no output at all.
    # Retry, and record the outcome in a file as well as on stdout -- the job's stdout can
    # be lost when SLURM kills it at the hard limit, which leaves no trace of what happened.
    CHAINLOG="snapshot/$EXP/chain.log"
    for attempt in 1 2 3; do
      NEXT=$(sbatch --parsable --job-name="$SLURM_JOB_NAME" \
             scripts/slurm_train_indoor.sh "$TRAIN_CONFIG" 2>&1)
      if [ $? -eq 0 ] && [ -n "$NEXT" ]; then
        echo "$(date -Is) $SLURM_JOB_ID -> successor $NEXT" | tee -a "$CHAINLOG"
        break
      fi
      echo "$(date -Is) $SLURM_JOB_ID attempt $attempt failed: $NEXT" | tee -a "$CHAINLOG" >&2
      [ "$attempt" = 3 ] && echo "WARNING: no successor queued, resubmit by hand" \
        | tee -a "$CHAINLOG" >&2
      sleep 20
    done
  fi
  kill "$PY" 2>/dev/null
  exit 0
}
trap on_time_limit USR1

python -u main.py "$TRAIN_CONFIG" &
PY=$!
wait "$PY"
