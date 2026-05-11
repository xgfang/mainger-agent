#!/bin/bash
#SBATCH --job-name=mainger-agent-test
#SBATCH --account=<your-allocation>
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=sft/logs/agent_test_%j.out
#SBATCH --error=sft/logs/agent_test_%j.err

# End-to-end smoke test: spin up vLLM serving the merged FT model, then run
# agent.py once via sft/test_agent_ft.py with --scaffold and verify that
# final.json + report.md + analysis.R + explanation.md all get produced.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PROJECT="$(pwd)"

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

SCRATCH_DIR=/scratch/<your-allocation>/$USER
export HF_HOME="$SCRATCH_DIR/hf-cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

MERGED="$SCRATCH_DIR/merged-mainger-qwen15b"
VLLM_LOG="sft/logs/agent_test_vllm_${SLURM_JOB_ID}.log"

echo "=== End-to-end agent.py + scaffold test ==="
echo "PROJECT : $PROJECT"
echo "MERGED  : $MERGED"
echo "==========================================="

# --------------------------------------------------------------------------- #
# Spin up vLLM in background                                                   #
# --------------------------------------------------------------------------- #
mkdir -p sft/logs
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

echo "[1/3] Starting vLLM ..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED" \
    --served-model-name qwen-1.5b-ft \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype bfloat16 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "    vLLM pid: $VLLM_PID"

echo "[2/3] Waiting for vLLM to come up ..."
READY=0
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
        echo "    vLLM ready after ${i}*5s"
        READY=1
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM died during startup. Last 50 lines:"
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: vLLM did not become ready in 5 minutes."
    tail -50 "$VLLM_LOG"
    exit 1
fi

# --------------------------------------------------------------------------- #
# Run the agent.py smoke test                                                  #
# --------------------------------------------------------------------------- #
echo "[3/3] Running sft/test_agent_ft.py ..."
python sft/test_agent_ft.py
EXIT_CODE=$?

echo "==========================================="
if [ "$EXIT_CODE" -eq 0 ]; then
    echo "PASS: agent.py with --scaffold produced final JSON artifacts."
    echo "Outputs in runs/agent_ft_smoke/:"
    ls -la runs/agent_ft_smoke/
else
    echo "FAIL: agent.py did not produce final JSON. Inspect:"
    echo "  runs/agent_ft_smoke/trace.json"
    echo "  $VLLM_LOG"
fi
echo "==========================================="

exit $EXIT_CODE
