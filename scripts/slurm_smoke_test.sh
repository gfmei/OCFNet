#!/bin/bash
#SBATCH --job-name=ocfnet_smoke
#SBATCH --time=00:20:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
#SBATCH --account=IscrC_ERAR --partition=boost_usr_prod
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source /leonardo_scratch/fast/AIFPT_agrifood/miniforge3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-reg3d}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /leonardo_scratch/fast/AIFPT_agrifood/code/OCFNet

# forward/backward of the spconv backbone, plus one real training iteration on 3DMatch
python -u scripts/test_spconv_forward.py configs/train/indoor.yaml
python -u scripts/test_train_step.py configs/train/indoor.yaml --iters 3
