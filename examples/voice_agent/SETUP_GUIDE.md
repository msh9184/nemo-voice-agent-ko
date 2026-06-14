# NeMo Voice Agent 설치 및 실행 가이드

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Server (A100 80GB)                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Voice Agent Server                     │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │   │
│  │  │   STT   │→ │  Diar   │→ │   LLM   │→ │     TTS     │ │   │
│  │  │Parakeet │  │Sortform │  │Nemotron │  │   Kokoro    │ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│              HTTP: 7860   │   WebSocket: 8765                  │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                     Network (LAN)
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                    Windows PC (Client)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            Web Browser (Chrome)                          │   │
│  │         http://[GPU_SERVER_IP]:5173                      │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │   │
│  │  │ Microphone│  │  Speaker  │  │   RTVI Client     │    │   │
│  │  └───────────┘  └───────────┘  └───────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 포트 정보

| 포트 | 용도 | 설명 |
|------|------|------|
| 7860 | FastAPI HTTP | /connect 엔드포인트 (WebSocket URL 반환) |
| 8765 | WebSocket | 실시간 오디오 스트리밍 |
| 5173 | Vite Dev Server | 웹 클라이언트 호스팅 |

---

## Step 1: GPU 서버 환경 준비

### 1.1 Conda 환경 생성

```bash
# GPU 서버에서 실행
cd /path/to/workspace

# Conda 환경 생성 (권장)
conda env create -f environment.yaml

# 환경 활성화
conda activate nemo-voice
```

### 1.2 또는 pip로 설치

```bash
# 기존 환경에서 pip 설치
pip install -r requirements.txt
```

### 1.3 필수 시스템 의존성 (Ubuntu/Debian)

```bash
# Node.js 설치 (클라이언트 빌드용)
curl -fsSL https://fnm.vercel.app/install | bash
source ~/.bashrc
fnm use --install-if-missing 20

# 또는 apt로 설치
sudo apt-get update
sudo apt-get install -y npm nodejs

# espeak-ng 설치 (Kokoro TTS용)
sudo apt-get install -y espeak-ng
```

---

## Step 2: 모델 다운로드 및 캐시 설정

### 2.1 환경 변수 설정

```bash
# ~/.bashrc 또는 스크립트에 추가

# NeMo 경로
export NEMO_PATH=/path/to/workspace
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH

# HuggingFace 캐시 경로 (네트워크 제한 환경용)
export HF_HUB_CACHE="/path/to/workspace"
export TRANSFORMERS_CACHE="/path/to/workspace"

# HuggingFace 토큰 (필요시)
# export HF_TOKEN="hf_your_token_here"

# 프록시 설정 (필요시)
# export HTTP_PROXY="http://proxy.company.com:port"
# export HTTPS_PROXY="http://proxy.company.com:port"
```

### 2.2 모델 사전 다운로드 (선택사항, 네트워크 제한 환경용)

프록시/방화벽 문제가 있는 경우, 네트워크가 원활한 환경에서 미리 다운로드:

```bash
# ASR 모델
huggingface-cli download nvidia/parakeet_realtime_eou_120m-v1 \
    --local-dir /path/to/workspace

# LLM 모델 (9B - A100 80GB에서 여유롭게 실행)
huggingface-cli download nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --local-dir /path/to/workspace

# TTS 모델
huggingface-cli download hexgrad/Kokoro-82M \
    --local-dir /path/to/workspace

# Speaker Diarization 모델
huggingface-cli download nvidia/diar_streaming_sortformer_4spk-v2 \
    --local-dir /path/to/workspace
```

### 2.3 VRAM 사용량 예측

| 컴포넌트 | 모델 | 예상 VRAM |
|----------|------|-----------|
| LLM | Nemotron-Nano-9B-v2 | ~18-21GB |
| STT | Parakeet-120M | ~1GB |
| TTS | Kokoro-82M | ~0.5GB |
| Diar | Sortformer-4spk | ~1GB |
| **Total** | | **~21-24GB** |

A100 80GB에서 충분한 여유 있음 ✓

---

## Step 3: 서버 설정 파일 구성

### 3.1 기본 설정 확인/수정

`server/server_configs/default.yaml` 파일을 환경에 맞게 수정:

```yaml
# default.yaml - 주요 설정 항목

transport:
  audio_out_10ms_chunks: 10  # TTS 출력 청크 크기

vad:
  type: silero
  confidence: 0.6      # VAD 신뢰도 임계값
  start_secs: 0.1      # 발화 시작 감지 시간
  stop_secs: 1.2       # 발화 종료 감지 시간 (조절 가능)
  min_volume: 0.4      # 최소 볼륨 임계값

stt:
  type: nemo
  # HuggingFace에서 직접 로드
  model: "nvidia/parakeet_realtime_eou_120m-v1"
  # 또는 로컬 경로 사용 (네트워크 문제 시)
  # model: "/path/to/workspace"
  model_config: "./server_configs/stt_configs/nemo_cache_aware_streaming.yaml"
  device: "cuda"

diar:
  type: nemo
  enabled: true        # false로 설정하면 화자 구분 비활성화
  model: "nvidia/diar_streaming_sortformer_4spk-v2"
  device: "cuda"
  threshold: 0.4

turn_taking:
  backchannel_phrases_path: "./server/backchannel_phrases.yaml"
  max_buffer_size: 2
  bot_stop_delay: 0.5

llm:
  type: auto           # vllm 또는 hf로 명시 가능
  model: "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
  # 로컬 경로 사용 시
  # model: "/path/to/workspace"
  model_config: "./server_configs/llm_configs/nemotron_nano_v2.yaml"
  device: "cuda"
  enable_reasoning: false  # true로 설정하면 reasoning 모드 활성화 (레이턴시 증가)
  system_prompt: "You are a helpful AI agent named Lisa..."

tts:
  type: kokoro
  model: "hexgrad/Kokoro-82M"
  model_config: "./server_configs/tts_configs/kokoro_82M.yaml"
  device: "cuda"
```

### 3.2 로컬 모델 경로 사용 시 설정 예시

네트워크 문제로 HuggingFace 직접 로드가 안 되는 경우:

```yaml
stt:
  model: "/path/to/workspace"

diar:
  model: "/path/to/workspace"

llm:
  model: "/path/to/workspace"

tts:
  model: "/path/to/workspace"
```

### 3.3 System Prompt 커스터마이징

```yaml
llm:
  # 인라인 프롬프트
  system_prompt: "당신은 친절한 AI 비서입니다. 한국어로 대화해 주세요."

  # 또는 파일 경로
  # system_prompt: "./server/example_prompts/my_custom_prompt.txt"
```

---

## Step 4: 클라이언트 설정 (원격 접속용)

### 4.1 클라이언트 소스 이해

`client/src/app.ts`에서 서버 URL 자동 감지:

```typescript
private readonly serverConfigs = {
  websocket: {
    name: 'WebSocket Server',
    baseUrl: `http://${window.location.hostname}:7860`,  // 브라우저 호스트명 자동 사용
    port: 8765
  },
  ...
};
```

브라우저에서 `http://GPU_SERVER_IP:5173`로 접속하면 자동으로 GPU 서버의 IP를 사용합니다.

### 4.2 방화벽/포트 확인 (GPU 서버)

```bash
# 포트 열림 확인
sudo netstat -tlnp | grep -E '7860|8765|5173'

# 방화벽 규칙 확인 (필요시)
sudo iptables -L -n | grep -E '7860|8765|5173'

# 포트 개방 (필요시)
sudo ufw allow 7860/tcp
sudo ufw allow 8765/tcp
sudo ufw allow 5173/tcp
```

### 4.3 Chrome 브라우저 설정 (Windows PC)

마이크 접근을 위해 insecure origin 허용 필요:

1. Chrome 주소창에 입력: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. 텍스트 박스에 추가: `http://GPU_SERVER_IP:5173`
3. **Enabled** 선택
4. **Relaunch** 클릭

---

## Step 5: 서버 실행

### 5.1 터미널 1: Voice Agent 서버 실행

```bash
# GPU 서버에서 실행
cd /path/to/workspace

# 환경 변수 설정
export NEMO_PATH=/path/to/workspace
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH
export HF_HUB_CACHE="/path/to/workspace"

# 커스텀 설정 파일 사용 시 (선택사항)
# export SERVER_CONFIG_PATH="/path/to/your/custom_config.yaml"

# Conda 환경 활성화
conda activate nemo-voice

# 서버 실행
python ./server/server.py
```

### 5.2 서버 시작 로그 확인

정상 시작 시 다음과 같은 로그가 출력됩니다:

```
INFO | Initializing STT service...
INFO | STT service initialized
INFO | Diarization service initialized
INFO | Turn taking service initialized
INFO | Initializing LLM service...
INFO | LLM service initialized
INFO | TTS service initialized
INFO | Setting up pipeline...
INFO | Starting pipeline runner...
INFO | Starting websocket server on 0.0.0.0:8765
INFO | Server configured to run indefinitely with no timeouts, use Ctrl+C to quit.
```

### 5.3 터미널 2: Web 클라이언트 실행

```bash
# GPU 서버에서 실행 (같은 서버)
cd /path/to/workspace

# Node 모듈 설치 (최초 1회)
npm install

# 개발 서버 실행
npm run dev
```

출력 예시:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: http://GPU_SERVER_IP:5173/
```

---

## Step 6: 클라이언트 접속 (Windows PC)

### 6.1 브라우저에서 접속

1. Chrome 브라우저 열기
2. 주소 입력: `http://GPU_SERVER_IP:5173`
3. **Connect** 버튼 클릭
4. 마이크 권한 허용

### 6.2 UI 기능

| 버튼 | 기능 |
|------|------|
| Connect | 서버에 연결 |
| Disconnect | 연결 해제 |
| Mute/Unmute | 마이크 음소거 토글 |
| Reset | 대화 컨텍스트 초기화 |

### 6.3 대화 테스트

1. Connect 후 "Connected" 상태 확인
2. 마이크에 대고 말하기
3. Debug Log에서 실시간 전사 확인:
   - `User: [사용자 발화]`
   - `Bot: [AI 응답]`
4. 스피커에서 TTS 음성 출력 확인

---

## 트러블슈팅

### 문제 1: HuggingFace 다운로드 실패

```
Error: HTTPConnectionPool(host='huggingface.co', port=443): ...
```

**해결책**: 로컬 모델 경로 사용
```yaml
# default.yaml에서 model 경로를 로컬로 변경
llm:
  model: "/path/to/workspace"
```

### 문제 2: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**해결책**: 더 작은 LLM 모델 사용
```yaml
llm:
  model: "nvidia/Nemotron-Mini-4B-Instruct"  # 13GB VRAM
```

### 문제 3: WebSocket 연결 실패

```
Error connecting: Cannot read properties of undefined
```

**해결책**:
1. 포트 7860, 8765 방화벽 확인
2. Chrome insecure origin 설정 확인
3. 서버 로그에서 에러 확인

### 문제 4: 마이크 권한 없음

```
Error: Permission denied
```

**해결책**:
1. `chrome://settings/content/microphone`에서 권한 확인
2. insecure origin 설정 확인 (Step 4.3)

### 문제 5: npm 설치 오류

```
SyntaxError: Unexpected reserved word
```

**해결책**:
```bash
rm -rf client/node_modules
fnm use 20  # Node.js 버전 업그레이드
cd client && npm install
```

### 문제 6: espeak-ng 오류 (Kokoro TTS)

```
RuntimeError: espeak-ng not found
```

**해결책**:
```bash
sudo apt-get install -y espeak-ng
```

---

## 성능 최적화 팁

### VAD 파라미터 조정

발화 종료 감지가 너무 빠르거나 느린 경우:

```yaml
vad:
  stop_secs: 0.8   # 더 빠른 응답 (기본 1.2)
  # stop_secs: 1.5 # 더 긴 대기 (끊김 방지)
```

### LLM 응답 속도 개선

```yaml
llm:
  type: vllm              # HuggingFace보다 빠름
  enable_reasoning: false  # reasoning 비활성화로 레이턴시 감소
```

### 화자 구분 비활성화 (단일 화자)

```yaml
diar:
  enabled: false  # VRAM 절약 및 레이턴시 감소
```

---

## 참고 명령어 모음

```bash
# 서버 백그라운드 실행 (nohup)
nohup python ./server/server.py > server.log 2>&1 &

# 서버 프로세스 확인
ps aux | grep server.py

# 서버 로그 확인
tail -f bot_server.log

# GPU 사용량 모니터링
watch -n 1 nvidia-smi

# 포트 사용 확인
netstat -tlnp | grep -E '7860|8765|5173'
```
