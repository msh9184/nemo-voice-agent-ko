#!/bin/bash
# =============================================================================
# Voice Agent Runner Script
# =============================================================================
#
# Usage:
#   ./run_voice_agent.sh [OPTIONS]
#
# Options:
#   -c, --config <path>     Path to server config YAML file
#   -m, --mode <mode>       Shortcut for predefined modes:
#                           stt-only, llm-only, tts-only,
#                           stt-llm, llm-tts, full (stt-llm-tts)
#   --server-only           Run only the server (no client)
#   --client-only           Run only the client (no server)
#   --build-client          Build client before running
#   --client-port <port>    Client dev server port (default: 5173)
#   --server-port <port>    WebSocket server port (default: 8765)
#   --api-port <port>       FastAPI server port (default: 7860)
#   -l, --list-configs      List available config files
#   -h, --help              Show this help message
#
# Examples:
#   ./run_voice_agent.sh -m stt-only
#   ./run_voice_agent.sh -c ./server/server_configs/korean_voice_agent_fish_speech_api.yaml
#   ./run_voice_agent.sh --server-only -m full
#   ./run_voice_agent.sh --build-client -m stt-llm-tts
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
CONFIG_PATH=""
RUN_SERVER=true
RUN_CLIENT=true
BUILD_CLIENT=false
CLIENT_PORT=5173
SERVER_PORT=8765
API_PORT=7860

# Config directory
CONFIG_DIR="./server/server_configs"

# PID file for tracking processes
PID_FILE="/tmp/voice_agent_pids.txt"

# =============================================================================
# Functions
# =============================================================================

show_help() {
    head -40 "$0" | tail -36
    exit 0
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

list_configs() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Available Configuration Files${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${GREEN}Mode Shortcuts (-m/--mode):${NC}"
    echo "  stt-only      → STT-Only Mode (Audio → Text)"
    echo "  llm-only      → LLM-Only Mode (Text → Text)"
    echo "  tts-only      → TTS-Only Mode (Text → Audio)"
    echo "  stt-llm       → STT+LLM Mode (Audio → Text → Response)"
    echo "  llm-tts       → LLM+TTS Mode (Text → Response → Audio)"
    echo "  full          → Full Mode (Audio → Text → Response → Audio)"
    echo ""
    echo -e "${GREEN}Available Config Files:${NC}"
    if [ -d "$CONFIG_DIR" ]; then
        for f in "$CONFIG_DIR"/*.yaml; do
            if [ -f "$f" ]; then
                basename "$f"
            fi
        done
    else
        echo "  Config directory not found: $CONFIG_DIR"
    fi
    echo ""
    echo -e "${GREEN}Korean Voice Agent Configs:${NC}"
    if [ -d "$CONFIG_DIR" ]; then
        ls "$CONFIG_DIR"/korean_voice_agent*.yaml 2>/dev/null | xargs -I {} basename {} || echo "  (none)"
    fi
    echo ""
    exit 0
}

get_config_for_mode() {
    local mode=$1
    case $mode in
        stt-only|stt_only)
            echo "$CONFIG_DIR/stt_only_mode.yaml"
            ;;
        llm-only|llm_only)
            echo "$CONFIG_DIR/llm_only_mode.yaml"
            ;;
        tts-only|tts_only)
            echo "$CONFIG_DIR/tts_only_mode.yaml"
            ;;
        stt-llm|stt_llm)
            echo "$CONFIG_DIR/stt_llm_mode.yaml"
            ;;
        llm-tts|llm_tts)
            echo "$CONFIG_DIR/llm_tts_mode.yaml"
            ;;
        full|stt-llm-tts|stt_llm_tts)
            echo "$CONFIG_DIR/stt_llm_tts_mode.yaml"
            ;;
        default)
            echo "$CONFIG_DIR/default.yaml"
            ;;
        *)
            log_error "Unknown mode: $mode"
            log_info "Available modes: stt-only, llm-only, tts-only, stt-llm, llm-tts, full"
            exit 1
            ;;
    esac
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check npm (only if running client)
    if [ "$RUN_CLIENT" = true ]; then
        if ! command -v npm &> /dev/null; then
            log_error "npm is not installed or not in PATH"
            exit 1
        fi
    fi

    # Check config file exists
    if [ -n "$CONFIG_PATH" ] && [ ! -f "$CONFIG_PATH" ]; then
        log_error "Config file not found: $CONFIG_PATH"
        exit 1
    fi

    log_success "Dependencies OK"
}

setup_environment() {
    log_info "Setting up environment..."

    # Set NEMO_PATH if not set (auto-detect)
    if [ -z "$NEMO_PATH" ]; then
        # Try to find NeMo root (2 levels up from examples/voice_agent)
        NEMO_PATH="$(cd "$SCRIPT_DIR/../.." && pwd)"
        export NEMO_PATH
        log_info "NEMO_PATH set to: $NEMO_PATH"
    fi

    # Add NeMo to PYTHONPATH
    export PYTHONPATH="${NEMO_PATH}:${PYTHONPATH:-}"

    # Set config path if specified
    if [ -n "$CONFIG_PATH" ]; then
        # Convert to absolute path if relative
        if [[ ! "$CONFIG_PATH" = /* ]]; then
            CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
        fi
        export SERVER_CONFIG_PATH="$CONFIG_PATH"
        log_info "SERVER_CONFIG_PATH: $CONFIG_PATH"
    fi

    log_success "Environment setup complete"
}

cleanup() {
    log_info "Cleaning up..."

    if [ -f "$PID_FILE" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                log_info "Stopping process $pid"
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi

    log_success "Cleanup complete"
}

# Trap to cleanup on exit
trap cleanup EXIT INT TERM

start_server() {
    log_info "Starting Voice Agent server..."

    cd "$SCRIPT_DIR/server"

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Voice Agent Server${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "  Config:     ${GREEN}${CONFIG_PATH:-default.yaml}${NC}"
    echo -e "  API Port:   ${GREEN}${API_PORT}${NC}"
    echo -e "  WS Port:    ${GREEN}${SERVER_PORT}${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Run server in foreground if client is not running, otherwise background
    if [ "$RUN_CLIENT" = false ]; then
        python server.py
    else
        python server.py &
        SERVER_PID=$!
        echo "$SERVER_PID" >> "$PID_FILE"
        log_success "Server started (PID: $SERVER_PID)"

        # Wait for server to be ready
        log_info "Waiting for server to be ready..."
        sleep 5
    fi
}

start_client() {
    log_info "Starting Voice Agent client..."

    cd "$SCRIPT_DIR/client"

    # Build if requested
    if [ "$BUILD_CLIENT" = true ]; then
        log_info "Building client..."
        npm run build
        log_success "Client build complete"
    fi

    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log_warn "node_modules not found, running npm install..."
        npm install
    fi

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Voice Agent Client${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "  URL:        ${GREEN}http://localhost:${CLIENT_PORT}${NC}"
    echo -e "  Mode:       ${GREEN}Development${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # Run client (this will block)
    npm run dev -- --port "$CLIENT_PORT" --host
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -m|--mode)
            CONFIG_PATH=$(get_config_for_mode "$2")
            shift 2
            ;;
        --server-only)
            RUN_CLIENT=false
            shift
            ;;
        --client-only)
            RUN_SERVER=false
            shift
            ;;
        --build-client)
            BUILD_CLIENT=true
            shift
            ;;
        --client-port)
            CLIENT_PORT="$2"
            shift 2
            ;;
        --server-port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        -l|--list-configs)
            list_configs
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# =============================================================================
# Main
# =============================================================================

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║             NeMo Voice Agent - Starting...                    ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

check_dependencies
setup_environment

# Clear previous PID file
rm -f "$PID_FILE"

# Start components
if [ "$RUN_SERVER" = true ]; then
    start_server
fi

if [ "$RUN_CLIENT" = true ]; then
    start_client
fi

# If only server is running, keep script alive
if [ "$RUN_SERVER" = true ] && [ "$RUN_CLIENT" = false ]; then
    log_info "Server running. Press Ctrl+C to stop."
    wait
fi
