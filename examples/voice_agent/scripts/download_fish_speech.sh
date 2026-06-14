#!/bin/bash
# =============================================================================
# OpenAudio S1-mini Model Download Script
# =============================================================================
# Downloads OpenAudio S1-mini model for Korean TTS
#
# Usage:
#   ./download_fish_speech.sh [model_dir]
#
# Default model directory: /path/to/model
# =============================================================================

set -e

# Configuration
MODEL_DIR="${1:-/path/to/model}"
MODEL_NAME="fishaudio/openaudio-s1-mini"

echo "=============================================="
echo "OpenAudio S1-mini Model Download Script"
echo "=============================================="
echo ""
echo "Model: ${MODEL_NAME}"
echo "Target directory: ${MODEL_DIR}"
echo ""
echo "OpenAudio S1-mini Features:"
echo "  - 0.5B parameters (distilled from 4B S1)"
echo "  - 2M+ hours training data with RLHF"
echo "  - 13 languages including Korean"
echo "  - 45+ emotion markers support"
echo ""

# Create directory
mkdir -p "${MODEL_DIR}"

# Download model using hf command (preferred)
echo "Downloading model..."
echo "This may take a while depending on your network speed."
echo ""

# Using hf command (already installed in most environments)
hf download ${MODEL_NAME} --local-dir "${MODEL_DIR}"

# Verify download
echo ""
echo "=============================================="
echo "Verifying downloaded files..."
echo "=============================================="

if [ -f "${MODEL_DIR}/codec.pth" ]; then
    echo "✅ codec.pth found"
else
    echo "❌ codec.pth NOT found - download may have failed"
fi

# List downloaded files
echo ""
echo "Downloaded files:"
ls -la "${MODEL_DIR}/"

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="
echo ""
echo "Model location: ${MODEL_DIR}"
echo ""
echo "Next steps:"
echo "  1. Install Fish Speech dependencies (if not using subprocess fallback)"
echo "     git clone https://github.com/fishaudio/fish-speech.git"
echo "     cd fish-speech && pip install -e .[cu129]"
echo ""
echo "  2. Run test script:"
echo "     python test_fish_speech_korean.py --model ${MODEL_DIR}"
echo ""
echo "  3. Start Voice Agent server:"
echo "     SERVER_CONFIG_PATH=./server_configs/korean_voice_agent_fish_speech.yaml python server.py"
echo ""
