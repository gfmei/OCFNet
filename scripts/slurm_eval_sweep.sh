#!/bin/bash
#SBATCH --job-name=eval_sweep
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:1 --mem=64G
#SBATCH --account=IscrC_ERAR --partition=boost_usr_prod
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source /leonardo_scratch/fast/AIFPT_agrifood/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-reg3d}"
# The evaluation is RANSAC-bound and parallelises over pairs, one thread each
# (scripts/evaluate_ocfnet.py); throttling OMP here would cap that instead.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /leonardo_scratch/fast/AIFPT_agrifood/code/OCFNet

CONFIG="${1:-${CONFIG:-configs/test/indoor.yaml}}"
test -f "$CONFIG" || { echo "no such config: $CONFIG" >&2; exit 2; }
BENCH="${BENCH:-3DMatch}"
SAMPLE_LIST="${SAMPLE_LIST:-5000 2500 1000 500 250}"
EXP=$(grep -E '^\s+exp_dir:' "$CONFIG" | awk '{print $2}')
TMP=/leonardo_scratch/fast/AIFPT_agrifood/tmp
mkdir -p "$TMP"

# The per-pair features do not depend on the number of sampled correspondences, so the
# expensive inference pass runs once and every sample count reuses the same dump.
RUN_CONFIG="$TMP/$(basename "${CONFIG%.yaml}")_sweep_$BENCH.yaml"
sed -e "s|^  benchmark: .*|  benchmark: $BENCH|" "$CONFIG" > "$RUN_CONFIG"
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
