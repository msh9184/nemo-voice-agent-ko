#!/bin/bash
# =============================================================================
# Voice Agent Stop Script
# =============================================================================
#
# Usage:
#   ./stop_voice_agent.sh [OPTIONS]
#
# Options:
#   --all           Stop all Voice Agent related processes
#   --server        Stop server processes only
#   --client        Stop client processes only
#   -f, --force     Force kill (SIGKILL instead of SIGTERM)
#   -h, --help      Show this help message
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# PID file
PID_FILE="/tmp/voice_agent_pids.txt"

# Options
STOP_ALL=true
STOP_SERVER=false
STOP_CLIENT=false
FORCE_KILL=false

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

show_help() {
    head -17 "$0" | tail -14
    exit 0
}

kill_process() {
    local pid=$1
    local name=$2

    if kill -0 "$pid" 2>/dev/null; then
        if [ "$FORCE_KILL" = true ]; then
            kill -9 "$pid" 2>/dev/null && log_info "Force killed $name (PID: $pid)"
        else
            kill "$pid" 2>/dev/null && log_info "Stopped $name (PID: $pid)"
        fi
    fi
}

stop_by_pattern() {
    local pattern=$1
    local name=$2

    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            kill_process "$pid" "$name"
        done
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            STOP_ALL=true
            shift
            ;;
        --server)
            STOP_ALL=false
            STOP_SERVER=true
            shift
            ;;
        --client)
            STOP_ALL=false
            STOP_CLIENT=true
            shift
            ;;
        -f|--force)
            FORCE_KILL=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

echo ""
echo -e "${YELLOW}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║             NeMo Voice Agent - Stopping...                    ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Stop from PID file first
if [ -f "$PID_FILE" ]; then
    log_info "Stopping processes from PID file..."
    while read -r pid; do
        kill_process "$pid" "Voice Agent process"
    done < "$PID_FILE"
    rm -f "$PID_FILE"
fi

# Stop server processes
if [ "$STOP_ALL" = true ] || [ "$STOP_SERVER" = true ]; then
    log_info "Stopping server processes..."
    stop_by_pattern "python.*server.py" "Voice Agent Server"
    stop_by_pattern "bot_websocket_server" "WebSocket Server"
    stop_by_pattern "uvicorn.*7860" "FastAPI Server"
fi

# Stop client processes
if [ "$STOP_ALL" = true ] || [ "$STOP_CLIENT" = true ]; then
    log_info "Stopping client processes..."
    stop_by_pattern "vite.*5173" "Vite Dev Server"
    stop_by_pattern "node.*vite" "Node Vite"
fi

echo ""
log_success "Voice Agent stopped"
echo ""

# Show remaining processes
remaining=$(pgrep -f "voice_agent|server.py|vite" 2>/dev/null || true)
if [ -n "$remaining" ]; then
    log_warn "Some related processes may still be running:"
    ps aux | grep -E "voice_agent|server.py|vite" | grep -v grep || true
fi
