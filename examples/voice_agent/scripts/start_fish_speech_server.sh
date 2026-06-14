#!/bin/bash
# =============================================================================
# Fish Speech Local API Server Startup Script
# =============================================================================
# This script starts the Fish Speech API server for low-latency TTS.
# The model is loaded ONCE and stays in GPU memory for fast inference (~1-3s).
#
# Usage:
#   ./start_fish_speech_server.sh [port]
#
# Default port: 8080
# =============================================================================

set -e

# Configuration
FISH_SPEECH_DIR="${FISH_SPEECH_DIR:-/path/to/workspace}"
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
PORT="${1:-8080}"
HOST="0.0.0.0"

echo "=============================================="
echo "Fish Speech API Server Startup"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Fish Speech Dir: ${FISH_SPEECH_DIR}"
echo "  Model Path: ${MODEL_PATH}"
echo "  Server: ${HOST}:${PORT}"
echo ""

# Check if fish-speech directory exists
if [ ! -d "${FISH_SPEECH_DIR}" ]; then
    echo "ERROR: Fish Speech directory not found: ${FISH_SPEECH_DIR}"
    echo "Please clone it first:"
    echo "  git clone https://github.com/fishaudio/fish-speech.git ${FISH_SPEECH_DIR}"
    exit 1
fi

# Check if model exists
if [ ! -f "${MODEL_PATH}/codec.pth" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    echo "Please download it first:"
    echo "  hf download fishaudio/openaudio-s1-mini --local-dir ${MODEL_PATH}"
    exit 1
fi

# Set up environment
cd "${FISH_SPEECH_DIR}"
export PYTHONPATH="${FISH_SPEECH_DIR}:${PYTHONPATH}"

echo "Starting Fish Speech API Server..."
echo "First run will take ~5-10 minutes for model loading and JIT compilation."
echo "Subsequent TTS requests will complete in ~1-3 seconds."
echo ""
echo "API Endpoints:"
echo "  - Health: http://${HOST}:${PORT}/"
echo "  - Docs:   http://${HOST}:${PORT}/docs"
echo "  - TTS:    http://${HOST}:${PORT}/v1/tts (POST)"
echo ""
echo "Press Ctrl+C to stop the server."
echo "=============================================="

# Start the API server
# Note: --compile enables torch.compile for faster inference after warmup
python -m tools.api_server \
    --listen "${HOST}:${PORT}" \
    --llama-checkpoint-path "${MODEL_PATH}" \
    --decoder-checkpoint-path "${MODEL_PATH}/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --compile
