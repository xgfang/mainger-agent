#!/bin/bash
#SBATCH --job-name=mainger-sft
#SBATCH --account=<your-allocation>
#SBATCH --partition=spgpu                # spgpu (A40), gpu_mig (A100 slice), gpu (V100)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=sft/logs/sft_%j.out
#SBATCH --error=sft/logs/sft_%j.err

# ----------------------------------------------------------------------
# This script trains Qwen2.5-1.5B-Instruct + LoRA on verified mainger
# teacher traces. Submit from the project root:
#   sbatch sft/train_qwen.sbatch
# Monitor:
#   squeue -u $USER
#   tail -f sft/logs/sft_<jobid>.out
# ----------------------------------------------------------------------

set -euo pipefail

# Environment
module purge
module load python3.10-anaconda/2023.03

# Activate by prepending the env's bin to PATH directly. We do NOT use
# `conda activate`: under sbatch it sets CONDA_DEFAULT_ENV / CONDA_PREFIX
# but does not update PATH, so `python` keeps resolving to the module's
# system anaconda. Prepending PATH is what conda activate is supposed to
# do; doing it explicitly works around the shell-function issue.
ENV_PREFIX=/home/$USER/.conda/envs/mainger-sft
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "ERROR: env python not found at $ENV_PREFIX/bin/python"
    echo "       /home may not be exported to compute nodes."
    echo "       If so, recreate the env under \$SCRATCH_DIR/conda-envs/."
    exit 1
fi
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"

# Block ~/.local/lib/python3.10/site-packages from shadowing the env.
export PYTHONNOUSERSITE=1

# Fail fast if anything went wrong.
echo "=== env diagnostics ==="
echo "ENV_PREFIX        : $ENV_PREFIX"
echo "which python      : $(which python)"
echo "python --version  : $(python --version 2>&1)"
python -c "import sys; print('sys.prefix       :', sys.prefix)"
python -c "import torch; print('torch            :', torch.__version__, 'cuda:', torch.cuda.is_available())" \
    || { echo "ERROR: torch import failed"; exit 1; }
python -c "import datasets, transformers, trl, peft, accelerate" \
    || { echo "ERROR: env package import failed"; exit 1; }
echo "============================"

# Force HuggingFace to use the cache on /scratch (NOT /home; /home is small)
export HF_HOME=/scratch/<your-allocation>/$USER/hf-cache
mkdir -p "$HF_HOME"

# Compute nodes typically can't reach HuggingFace. Pre-download the base
# model on a login node before submitting (see GREATLAKES_SFT.md). With
# the cache populated, this flag forces transformers to never phone home:
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Paths
PROJECT=$(pwd)                   # assume submitted from project root
DATA=$PROJECT/sft/data/teacher_traces_complete.jsonl
OUTPUT=$PROJECT/sft/checkpoints/qwen-1.5b-mainger-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUTPUT" "$PROJECT/sft/logs"

# Confirm GPU is visible before starting (helps debugging)
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

# Train
python $PROJECT/sft/train_lora.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --data_path  "$DATA" \
    --output_dir "$OUTPUT" \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.1 \
    --max_seq_length 4096 \
    --logging_steps 10 \
    --save_strategy epoch \
    --eval_fraction 0.05

echo "Training done. Adapter at: $OUTPUT"
