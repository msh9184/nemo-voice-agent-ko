#!/bin/bash
# install_voice_agent.sh
# Voice Agent 설치 스크립트 (PyTorch 2.6.x 환경 유지)
#
# 사용법:
#   chmod +x install_voice_agent.sh
#   ./install_voice_agent.sh [--stt-only] [--full]
#
# 옵션:
#   --stt-only  : STT-only 모드 설치 (LLM/TTS 제외, 최소 설치)
#   --full      : 전체 Voice Agent 설치 (기본값)

set -e  # 에러 발생 시 즉시 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE} NeMo Voice Agent 설치 스크립트 v2${NC}"
echo -e "${BLUE} PyTorch 2.6.x 환경 유지${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 모드 설정
MODE="full"
if [[ "$1" == "--stt-only" ]]; then
    MODE="stt-only"
    echo -e "${GREEN}[INFO] STT-only 모드 설치${NC}"
elif [[ "$1" == "--full" ]]; then
    MODE="full"
    echo -e "${GREEN}[INFO] 전체 Voice Agent 설치${NC}"
else
    echo -e "${GREEN}[INFO] 기본 모드: 전체 Voice Agent 설치${NC}"
fi
echo ""

# Step 1: 현재 PyTorch 버전 확인
echo -e "${YELLOW}[Step 1/10] 현재 PyTorch 버전 확인...${NC}"
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "N/A")

echo -e "  PyTorch: ${GREEN}${PYTORCH_VERSION}${NC}"
echo -e "  CUDA:    ${GREEN}${CUDA_VERSION}${NC}"

if [[ "$PYTORCH_VERSION" == "NOT_INSTALLED" ]]; then
    echo -e "${RED}[ERROR] PyTorch가 설치되어 있지 않습니다.${NC}"
    exit 1
fi

# 원본 버전 저장
ORIGINAL_PYTORCH_VERSION="$PYTORCH_VERSION"

# PyTorch 2.6.x 확인
if [[ ! "$PYTORCH_VERSION" == 2.6* ]]; then
    echo -e "${YELLOW}[WARNING] PyTorch ${PYTORCH_VERSION}는 2.6.x가 아닙니다.${NC}"
    read -p "계속하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Step 2: NumPy 버전 고정 (2.0 미만)
echo -e "${YELLOW}[Step 2/10] NumPy 버전 확인 및 고정 (<2.0.0)...${NC}"
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
echo -e "  현재 NumPy: ${NUMPY_VERSION}"

if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo -e "  ${YELLOW}NumPy 2.x 감지됨. 1.x로 다운그레이드...${NC}"
    pip install "numpy<2.0.0" --force-reinstall
elif [[ "$NUMPY_VERSION" == "NOT_INSTALLED" ]]; then
    pip install "numpy<2.0.0"
else
    echo -e "  ${GREEN}NumPy 버전 OK${NC}"
fi
echo ""

# Step 3: 기존 vLLM 제거 (있다면)
echo -e "${YELLOW}[Step 3/10] 기존 vLLM 확인 및 제거...${NC}"
if pip show vllm &>/dev/null; then
    VLLM_VERSION=$(pip show vllm | grep "Version:" | cut -d' ' -f2)
    echo -e "  기존 vLLM 버전: ${VLLM_VERSION}"
    if [[ "$VLLM_VERSION" == 0.10* ]] || [[ "$VLLM_VERSION" == 0.9* ]]; then
        echo -e "  ${YELLOW}vLLM ${VLLM_VERSION}은 PyTorch 2.7+를 요구합니다. 제거합니다...${NC}"
        pip uninstall vllm -y
    fi
else
    echo -e "  ${GREEN}vLLM이 설치되어 있지 않습니다.${NC}"
fi
echo ""

# Step 4: Transformer Engine 업그레이드
echo -e "${YELLOW}[Step 4/10] Transformer Engine 설치/업그레이드...${NC}"
pip install --upgrade transformer_engine[pytorch] 2>/dev/null || {
    echo -e "  ${YELLOW}[WARNING] Transformer Engine 설치 실패 - 이미 최신 버전일 수 있습니다${NC}"
}
echo -e "  ${GREEN}Transformer Engine 완료${NC}"
echo ""

# Step 5: Flash Attention 설치
echo -e "${YELLOW}[Step 5/10] Flash Attention 설치...${NC}"
pip install flash-attn==2.8.0.post2 --no-build-isolation --no-deps --force-reinstall 2>/dev/null || {
    echo -e "  ${YELLOW}[WARNING] Flash Attention 설치 실패${NC}"
}
echo -e "  ${GREEN}Flash Attention 완료${NC}"
echo ""

# Step 6: NVIDIA Logger 및 Kaldi 설치
echo -e "${YELLOW}[Step 6/10] NVIDIA Logger 및 Kaldi Align 설치...${NC}"
pip install nv-one-logger-core nv-one-logger-training-telemetry nv-one-logger-pytorch-lightning-integration 2>/dev/null || {
    echo -e "  ${YELLOW}[WARNING] NVIDIA Logger 일부 패키지 설치 실패${NC}"
}
pip install kaldialign 2>/dev/null || {
    echo -e "  ${YELLOW}[WARNING] kaldialign 설치 실패${NC}"
}
echo -e "  ${GREEN}NVIDIA Logger 및 Kaldi 완료${NC}"
echo ""

# Step 7: Pipecat 의존성 설치 (정확한 버전)
echo -e "${YELLOW}[Step 7/10] pipecat-ai 의존성 설치 (정확한 버전)...${NC}"
echo -e "  ${BLUE}pipecat-ai 0.0.84 요구사항에 맞게 설치 중...${NC}"

# pipecat-ai 0.0.84의 정확한 의존성 버전 설치
pip install "aiofiles>=24.1.0,<25" --force-reinstall 2>/dev/null || true
pip install "aiohttp>=3.11.12,<4" --force-reinstall 2>/dev/null || true
pip install "Pillow>=11.1.0,<12" --force-reinstall 2>/dev/null || true
pip install "protobuf>=5.29.0,<6" --force-reinstall 2>/dev/null || true

# 기타 pipecat 의존성
pip install wait_for2 2>/dev/null || true
pip install "pydantic>=2.0.0" 2>/dev/null || true
pip install "loguru>=0.7.0" 2>/dev/null || true
pip install "websockets>=15.0.1" 2>/dev/null || true
pip install httpx typing_extensions 2>/dev/null || true
pip install numpy-quaternion scipy audioop-lts 2>/dev/null || true
pip install daily-python 2>/dev/null || true

echo -e "  ${GREEN}pipecat 의존성 설치 완료${NC}"
echo ""

# Step 8: Pipecat 설치
echo -e "${YELLOW}[Step 8/10] pipecat-ai 설치...${NC}"
pip install pipecat-ai==0.0.84 --no-deps
echo -e "  ${GREEN}pipecat-ai 설치 완료${NC}"
echo ""

# Step 9: LLM 설치 (full 모드인 경우)
if [[ "$MODE" == "full" ]]; then
    echo -e "${YELLOW}[Step 9/10] vLLM 0.8.5.post1 설치 (PyTorch 2.6.x 호환)...${NC}"
    echo -e "  ${BLUE}주의: vLLM 설치는 시간이 걸릴 수 있습니다...${NC}"

    # vLLM 0.8.5.post1 설치 (PyTorch 업그레이드 방지)
    pip install vllm==0.8.5.post1 --no-deps

    # vLLM 필수 의존성만 설치 (PyTorch 제외)
    pip install "ray>=2.9" "pyarrow>=12.0.0" "uvloop>=0.17.0" 2>/dev/null || true
    pip install "prometheus_client>=0.17.0" "py-cpuinfo>=9.0.0" 2>/dev/null || true
    pip install msgspec gguf compressed-tensors 2>/dev/null || true
    pip install "outlines>=0.0.43,<1" "lark>=1.1.0" "interegular>=0.3.2" 2>/dev/null || true
    pip install "xgrammar>=0.1.6" "partial_json_parser" 2>/dev/null || true

    echo -e "  ${GREEN}vLLM 설치 완료${NC}"
else
    echo -e "${YELLOW}[Step 9/10] STT-only 모드: vLLM 설치 건너뜀${NC}"
fi
echo ""

# Step 10: TTS 설치 (full 모드인 경우)
if [[ "$MODE" == "full" ]]; then
    echo -e "${YELLOW}[Step 10/10] TTS (Kokoro) 설치...${NC}"
    pip install kokoro==0.9.4 espeakng==1.0.3 2>/dev/null || {
        echo -e "  ${YELLOW}[WARNING] Kokoro TTS 설치 실패${NC}"
    }
    echo -e "  ${GREEN}TTS 설치 완료${NC}"
else
    echo -e "${YELLOW}[Step 10/10] STT-only 모드: TTS 설치 건너뜀${NC}"
fi
echo ""

# 최종 검증
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE} 설치 검증${NC}"
echo -e "${BLUE}============================================${NC}"

echo -e "${YELLOW}PyTorch 버전 확인...${NC}"
FINAL_PYTORCH=$(python -c "import torch; print(torch.__version__)")
echo -e "  PyTorch: ${GREEN}${FINAL_PYTORCH}${NC}"

if [[ ! "$FINAL_PYTORCH" == "$ORIGINAL_PYTORCH_VERSION" ]]; then
    echo -e "${RED}[ERROR] PyTorch 버전이 변경되었습니다!${NC}"
    echo -e "${RED}        원래: ${ORIGINAL_PYTORCH_VERSION} → 현재: ${FINAL_PYTORCH}${NC}"
else
    echo -e "  ${GREEN}✓ PyTorch 버전 유지 확인됨${NC}"
fi

echo ""
echo -e "${YELLOW}핵심 패키지 버전 확인...${NC}"
echo -e "  pipecat-ai:          $(pip show pipecat-ai 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
echo -e "  protobuf:            $(pip show protobuf 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
echo -e "  aiofiles:            $(pip show aiofiles 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
echo -e "  aiohttp:             $(pip show aiohttp 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
echo -e "  Pillow:              $(pip show Pillow 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
echo -e "  websockets:          $(pip show websockets 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
echo -e "  transformer_engine:  $(pip show transformer_engine 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
echo -e "  flash_attn:          $(pip show flash-attn 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"

if [[ "$MODE" == "full" ]]; then
    echo -e "  vllm:                $(pip show vllm 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
    echo -e "  kokoro:              $(pip show kokoro 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'NOT INSTALLED')"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN} 설치 완료!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${YELLOW}[주의] protobuf 5.x로 업그레이드됨 - NeMo 일부 기능에 영향이 있을 수 있습니다${NC}"
echo ""

if [[ "$MODE" == "stt-only" ]]; then
    echo -e "다음 명령어로 STT-only 서버를 시작하세요:"
    echo -e "  ${BLUE}cd /path/to/workspace"
    echo -e "  ${BLUE}export NEMO_PATH=/path/to/workspace"
    echo -e "  ${BLUE}export PYTHONPATH=\$NEMO_PATH:\$PYTHONPATH${NC}"
    echo -e "  ${BLUE}python ./server/stt_only_server.py${NC}"
else
    echo -e "다음 명령어로 Voice Agent 서버를 시작하세요:"
    echo -e "  ${BLUE}cd /path/to/workspace"
    echo -e "  ${BLUE}export NEMO_PATH=/path/to/workspace"
    echo -e "  ${BLUE}export PYTHONPATH=\$NEMO_PATH:\$PYTHONPATH${NC}"
    echo -e "  ${BLUE}python ./server/server.py${NC}"
fi
echo ""
