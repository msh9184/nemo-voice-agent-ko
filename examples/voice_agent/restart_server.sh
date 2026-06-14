#!/bin/bash
# NeMo Voice Agent - Server Restart Script
# GPU 서버용 서버 재시작 스크립트
#
# Usage:
#   ./restart_server.sh          # STT-only 모드 (기본)
#   ./restart_server.sh full     # Full Voice Agent 모드
#   ./restart_server.sh debug    # STT-only + DEBUG 로깅

set -e

MODE=${1:-stt}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}NeMo Voice Agent Server Restart${NC}"
echo -e "${YELLOW}========================================${NC}"

# Step 1: Stop existing servers
echo -e "\n${YELLOW}[1/4] Stopping existing servers...${NC}"
pkill -f "stt_only_server.py" 2>/dev/null || true
pkill -f "server.py" 2>/dev/null || true
sleep 2

# Step 2: Check and free ports
echo -e "${YELLOW}[2/4] Checking ports...${NC}"
for port in 7860 8765 8000; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo -e "${RED}Port $port still in use, force killing...${NC}"
        fuser -k $port/tcp 2>/dev/null || true
    else
        echo -e "${GREEN}Port $port is free${NC}"
    fi
done
sleep 1

# Step 3: Set environment variables
echo -e "${YELLOW}[3/4] Setting environment...${NC}"

# Detect base path (adjust if needed)
if [ -d "/path/to/workspace" ]; then
    NEMO_BASE="/path/to/workspace"
else
    NEMO_BASE="$SCRIPT_DIR/../.."
fi

export NEMO_PATH="$NEMO_BASE"
export PYTHONPATH="$NEMO_PATH:$PYTHONPATH"
export HF_HUB_CACHE="/path/to/workspace"
export TRANSFORMERS_CACHE="/path/to/workspace"

echo "NEMO_PATH=$NEMO_PATH"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

# Step 4: Start server
echo -e "${YELLOW}[4/4] Starting server...${NC}"
cd "$SCRIPT_DIR"

case "$MODE" in
    stt|stt_only)
        echo -e "${GREEN}Starting STT-only server...${NC}"
        unset LOG_LEVEL
        python server/stt_only_server.py
        ;;
    debug)
        echo -e "${GREEN}Starting STT-only server with DEBUG logging...${NC}"
        export LOG_LEVEL=DEBUG
        python server/stt_only_server.py
        ;;
    full|voice_agent)
        echo -e "${GREEN}Starting Full Voice Agent server...${NC}"
        unset LOG_LEVEL
        python server/server.py
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 [stt|debug|full]"
        exit 1
        ;;
esac
