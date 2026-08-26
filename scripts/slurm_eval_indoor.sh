#!/bin/bash
#SBATCH --job-name=ocfnet_eval
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:1 --mem=64G
#SBATCH --account=IscrC_ERAR --partition=boost_usr_prod
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Registration Recall / Feature Match Recall / Inlier Ratio on one benchmark.
#
#   sbatch --export=ALL,BENCH=3DMatch,SAMPLES=1000 scripts/slurm_eval_indoor.sh
#
# The dumped features are benchmark specific -- the test loader is built from `benchmark`
# in the config -- so every benchmark needs its own main.py run.
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
SAMPLES="${SAMPLES:-1000}"
EXP=$(grep -E '^\s+exp_dir:' "$CONFIG" | awk '{print $2}')
TMP=/leonardo_scratch/fast/AIFPT_agrifood/tmp
mkdir -p "$TMP"

# 1. per-point features and scores for every pair of this benchmark
RUN_CONFIG="$TMP/$(basename "${CONFIG%.yaml}")_$BENCH.yaml"
sed -e "s|^  benchmark: .*|  benchmark: $BENCH|" "$CONFIG" > "$RUN_CONFIG"
python -u main.py "$RUN_CONFIG"

# 2. RANSAC over the sampled correspondences, then the three metrics (the slow part).
#    OCFNet predicts correspondences, Predator predicts descriptors -- different scripts.
MODEL=$(grep -E '^\s+model:' "$CONFIG" | awk '{print $2}')
if [ "$MODEL" = "OCFNet" ]; then EVAL=scripts/evaluate_ocfnet.py; else EVAL=scripts/evaluate_predator.py; fi
echo "########## $BENCH, $SAMPLES samples, $MODEL -> $EVAL"
python -u "$EVAL" --source_path "snapshot/$EXP/$BENCH" \
    --n_points "$SAMPLES" --benchmark "$BENCH" --exp_dir "snapshot/$EXP/est_traj"

echo "########## results"
cat "snapshot/$EXP/est_traj/$BENCH/$SAMPLES/result" 2>/dev/null || true
