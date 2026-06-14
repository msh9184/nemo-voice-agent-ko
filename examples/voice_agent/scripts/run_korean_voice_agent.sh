#!/bin/bash
#
# Korean Voice Agent Server Startup Script
#
# This script starts the Korean Voice Agent server with the appropriate configuration.
#
# Usage:
#   ./run_korean_voice_agent.sh [melo|kokoro|cosyvoice3]
#
# Options:
#   melo       - Use MeloTTS Korean for TTS (default, CPU real-time)
#   kokoro     - Use Kokoro Korean for TTS (requires GPU)
#   cosyvoice3 - Use CosyVoice3 for TTS (best quality, requires GPU, ~7.5GB model)
#
# Prerequisites:
#   - Python 3.8+
#   - CUDA-capable GPU (for STT and LLM)
#   - Install dependencies:
#     - melo: pip install melotts
#     - kokoro: pip install kokoro>=0.9.2
#     - cosyvoice3: git clone https://github.com/FunAudioLLM/CosyVoice.git && cd CosyVoice && pip install -e .
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(dirname "$SCRIPT_DIR")/server"

# Default TTS type
TTS_TYPE="${1:-melo}"

# Select configuration based on TTS type
if [ "$TTS_TYPE" = "kokoro" ]; then
    CONFIG_FILE="./server_configs/korean_voice_agent_kokoro.yaml"
    echo "Using Kokoro Korean TTS configuration"
elif [ "$TTS_TYPE" = "cosyvoice3" ]; then
    CONFIG_FILE="./server_configs/korean_voice_agent_cosyvoice3.yaml"
    echo "Using CosyVoice3 Korean TTS configuration (best quality)"
else
    CONFIG_FILE="./server_configs/korean_voice_agent.yaml"
    echo "Using MeloTTS Korean configuration"
fi

# Change to server directory
cd "$SERVER_DIR"

# Print configuration
echo ""
echo "=========================================="
echo "Korean Voice Agent Server"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "Server directory: $SERVER_DIR"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Check required Python packages
echo "Checking Python dependencies..."

python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

if [ "$TTS_TYPE" = "kokoro" ]; then
    python3 -c "from kokoro import KPipeline; print('Kokoro: OK')" 2>/dev/null || {
        echo "WARNING: Kokoro not installed. Install with: pip install kokoro>=0.9.2"
    }
elif [ "$TTS_TYPE" = "cosyvoice3" ]; then
    python3 -c "from cosyvoice.cli.cosyvoice import CosyVoice2; print('CosyVoice3: OK')" 2>/dev/null || {
        python3 -c "from cosyvoice.cli.model import AutoModel; print('CosyVoice3: OK')" 2>/dev/null || {
            echo "WARNING: CosyVoice not installed. Install with:"
            echo "  git clone https://github.com/FunAudioLLM/CosyVoice.git"
            echo "  cd CosyVoice && pip install -e ."
        }
    }
else
    python3 -c "from melo.api import TTS; print('MeloTTS: OK')" 2>/dev/null || {
        echo "WARNING: MeloTTS not installed. Install with: pip install melotts"
    }
fi

python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "Starting Korean Voice Agent server..."
echo "Press Ctrl+C to stop"
echo ""

# Set environment variables
export SERVER_CONFIG_PATH="$CONFIG_FILE"
export WEBSOCKET_SERVER="websocket_server"

# Optional: Set specific CUDA device
# export CUDA_VISIBLE_DEVICES=0

# Run server
python3 server.py
