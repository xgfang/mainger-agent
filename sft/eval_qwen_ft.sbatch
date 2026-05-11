#!/bin/bash
#SBATCH --job-name=mainger-eval
#SBATCH --account=<your-allocation>
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=sft/logs/eval_%j.out
#SBATCH --error=sft/logs/eval_%j.err

# ---------------------------------------------------------------------------
# Pipeline:
#   1. Merge LoRA adapter into base, save merged model to scratch (one time).
#   2. Install vLLM if not already in env.
#   3. Spin up vLLM OpenAI-compatible server in background.
#   4. Wait for vLLM to be ready (poll /v1/models).
#   5. Run sim/run_llm_evaluation.py (S3 only) against the local vLLM.
#   6. Aggregate results.
#   7. Cleanup vLLM (trap on exit).
# ---------------------------------------------------------------------------

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PROJECT="$(pwd)"

# Activate env (pattern proven to work on compute nodes from training sbatch)
module purge
module load python3.10-anaconda/2023.03
module load R/4.4.0

ENV_PREFIX=/home/$USER/.conda/envs/mainger-sft
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    echo "ERROR: env python not found at $ENV_PREFIX/bin/python"
    exit 1
fi
export PATH="$ENV_PREFIX/bin:$PATH"
export CONDA_PREFIX="$ENV_PREFIX"
export PYTHONNOUSERSITE=1

# HF + scratch paths
SCRATCH_DIR=/scratch/<your-allocation>/$USER
export HF_HOME="$SCRATCH_DIR/hf-cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

ADAPTER="$PROJECT/sft/checkpoints/qwen-1.5b-mainger-20260508-080752"
MERGED="$SCRATCH_DIR/merged-mainger-qwen15b"
RESULTS="sim/data/llm_results_qwen15bft_${SLURM_JOB_ID}.jsonl"
SUMMARY="sft/logs/eval_summary_${SLURM_JOB_ID}.txt"

echo "=== Eval pipeline start ==="
echo "PROJECT  : $PROJECT"
echo "ADAPTER  : $ADAPTER"
echo "MERGED   : $MERGED"
echo "RESULTS  : $RESULTS"
echo "==========================="

# --------------------------------------------------------------------------- #
# Phase 1: merge adapter into base                                             #
# --------------------------------------------------------------------------- #
if [ ! -f "$MERGED/config.json" ]; then
    echo "[1/6] Merging LoRA adapter into base ..."
    python - <<PYEOF
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"Loading base model {base_id} in bf16 ...")
base = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True,
)
print("Loading adapter ...")
m = PeftModel.from_pretrained(base, "$ADAPTER")
print("Merging ...")
m = m.merge_and_unload()
print(f"Saving merged model to $MERGED ...")
m.save_pretrained("$MERGED", safe_serialization=True)
tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
tok.save_pretrained("$MERGED")
print("Merge complete.")
PYEOF
else
    echo "[1/6] Merged model already exists at $MERGED, skipping merge."
fi

# --------------------------------------------------------------------------- #
# Phase 2: install vLLM if needed                                              #
# --------------------------------------------------------------------------- #
if ! python -c "import vllm" 2>/dev/null; then
    echo "[2/6] Installing vLLM ..."
    pip install --quiet "vllm>=0.6,<0.7"
else
    echo "[2/6] vLLM already installed."
fi
python -c "import vllm; print(f'    vllm {vllm.__version__}')"

# --------------------------------------------------------------------------- #
# Phase 3: spin up vLLM in background                                          #
# --------------------------------------------------------------------------- #
echo "[3/6] Starting vLLM server in background ..."
mkdir -p sft/logs
VLLM_LOG="sft/logs/vllm_${SLURM_JOB_ID}.log"

# trap to ensure vLLM is killed when the sbatch exits (success or failure)
VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Cleanup: stopping vLLM (pid $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 3
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" \
    --served-model-name qwen-1.5b-ft \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype bfloat16 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "    vLLM pid: $VLLM_PID"
echo "    vLLM log: $VLLM_LOG"

# --------------------------------------------------------------------------- #
# Phase 4: wait for vLLM to be ready                                           #
# --------------------------------------------------------------------------- #
echo "[4/6] Waiting for vLLM to come up (up to 5 min) ..."
READY=0
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
        echo "    vLLM ready after ${i}*5s"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died during startup. Last 50 lines of log:"
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: vLLM did not become ready within 5 minutes. Last 50 lines:"
    tail -50 "$VLLM_LOG"
    exit 1
fi
curl -s http://127.0.0.1:8000/v1/models

# --------------------------------------------------------------------------- #
# Phase 5: run eval                                                            #
# --------------------------------------------------------------------------- #
echo "[5/6] Running S3 eval against vLLM ..."
export VLLM_URL=http://127.0.0.1:8000/v1
python sim/run_llm_evaluation.py \
    --problems sim/data/sim_problems_moderate.jsonl \
    --output "$RESULTS" \
    --strategies S3 \
    --models qwen-1.5b-ft \
    --rps 1.0

# --------------------------------------------------------------------------- #
# Phase 6: aggregate                                                           #
# --------------------------------------------------------------------------- #
echo "[6/6] Aggregating ..."
python sim/aggregate_results.py \
    --problems sim/data/sim_problems_moderate.jsonl \
    --results "$RESULTS" \
    > "$SUMMARY" 2>&1 || {
        echo "WARN: aggregate_results.py failed; raw results still saved at $RESULTS"
        cat "$SUMMARY" || true
    }

echo "=== Eval pipeline done ==="
echo "Raw results : $RESULTS"
echo "Summary     : $SUMMARY"
echo "vLLM log    : $VLLM_LOG"
