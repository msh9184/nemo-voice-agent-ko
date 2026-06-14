# NeMo Voice Agent - Troubleshooting Guide

## Quick Reference

| 증상 | 빠른 해결 | 상세 섹션 |
|------|----------|-----------|
| STT 출력 없음 | 서버 재시작 + 포트 확인 | [STT 미작동](#1-stt-출력이-나오지-않음) |
| WebSocket 연결 실패 | 포트 점유 확인 | [연결 문제](#2-websocket-연결-실패) |
| 마이크 권한 오류 | HTTPS 확인 + 브라우저 설정 | [마이크 문제](#3-마이크-권한-문제) |
| Mixed Content 오류 | Protocol auto-detection 확인 | [HTTPS 문제](#4-https-mixed-content-오류) |

---

## 1. STT 출력이 나오지 않음

### 증상
- WebSocket 연결 성공 ("RTVI Client ready" 로그 출력)
- 마이크 waveform 표시됨
- 하지만 transcription 결과가 나오지 않음

### 진단 순서

#### Step 1: 서버 프로세스 상태 확인
```bash
# 포트 점유 확인
lsof -i :7860
lsof -i :8765

# 또는
netstat -tlnp | grep -E '7860|8765'
```

#### Step 2: 좀비 프로세스 정리
```bash
# Python 프로세스 확인
ps aux | grep python | grep -E 'stt_only|server'

# 강제 종료 (PID 확인 후)
kill -9 <PID>

# 또는 한번에 정리
pkill -f "stt_only_server.py"
pkill -f "server.py"
```

#### Step 3: 서버 완전 재시작
```bash
# 환경 변수 초기화 후 재시작
unset LOG_LEVEL
python server/stt_only_server.py
```

#### Step 4: DEBUG 로깅 활성화 (문제 지속 시)
```bash
export LOG_LEVEL=DEBUG
python server/stt_only_server.py
```

확인할 로그:
```
[WS-DEBUG] Message #1: type=bytes, len=xxx
[WS-DEBUG] Audio frame #1: 1280 bytes, sample_rate=16000
```

### 해결 체크리스트
- [ ] 이전 서버 프로세스 종료 확인
- [ ] 포트 7860, 8765 사용 가능 확인
- [ ] 클라이언트 브라우저 새로고침 (Ctrl+Shift+R)
- [ ] 서버 재시작

---

## 2. WebSocket 연결 실패

### 증상
- "Connect" 버튼 클릭 후 연결되지 않음
- 브라우저 콘솔에 WebSocket 에러

### 진단

#### 서버 상태 확인
```bash
# 서버가 실행 중인지 확인
curl http://localhost:7860/health

# 예상 응답
# {"status": "healthy", "mode": "stt_only"}
```

#### 포트 접근성 확인
```bash
# WebSocket 포트 테스트
nc -zv localhost 8765
```

### 해결 방법

#### SSH 터널 사용 시 (원격 접속)
```bash
# 로컬 PC에서 실행
ssh -L 5173:localhost:5173 -L 7860:localhost:7860 -L 8765:localhost:8765 user@gpu-server
```

#### 방화벽 확인 (GPU 서버)
```bash
# 포트 열기 (필요시)
sudo ufw allow 7860
sudo ufw allow 8765
```

---

## 3. 마이크 권한 문제

### 증상
- Waveform이 표시되지 않음
- 브라우저에서 마이크 권한 요청 없음

### 진단: 브라우저 콘솔 확인 (F12)
```
NotAllowedError: Permission denied
```
또는
```
NotFoundError: Requested device not found
```

### 해결 방법

#### HTTPS 필수 (Chrome 정책)
HTTP에서는 마이크 접근이 제한됩니다. HTTPS 사용:

```javascript
// vite.config.js
export default defineConfig({
    server: {
        https: {
            key: fs.readFileSync('./key.pem'),
            cert: fs.readFileSync('./cert.pem'),
        },
    },
});
```

#### Self-signed 인증서 생성
```bash
cd examples/voice_agent/client
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

#### Chrome에서 localhost 예외 설정
1. `chrome://flags/#unsafely-treat-insecure-origin-as-secure` 접속
2. `http://localhost:5173` 추가
3. 브라우저 재시작

---

## 4. HTTPS Mixed Content 오류

### 증상
- HTTPS 페이지에서 ws:// 연결 시도 시 차단
- 브라우저 콘솔: "Mixed Content: ... was loaded over HTTPS, but attempted to connect to the insecure WebSocket endpoint"

### 해결: Protocol Auto-detection 확인

#### 클라이언트 (app.ts)
```typescript
private readonly httpProtocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
```

#### 서버 (stt_only_server.py)
```python
@app.post("/connect")
async def bot_connect(request: Request):
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    is_secure = request.url.scheme == "https" or forwarded_proto == "https"
    ws_protocol = "wss" if is_secure else "ws"
    # ...
```

### localhost 예외
- Chrome은 localhost에 대해 HTTP→WS 연결을 허용함
- 외부 호스트에서는 HTTPS→WSS 필수

---

## 5. 서버 시작 오류

### CUDA 관련 경고 (무시 가능)
```
Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
```
→ TensorFlow/JAX와 PyTorch 동시 로드 시 발생, 동작에는 영향 없음

### 모델 로딩 실패
```bash
# HuggingFace 캐시 경로 설정
export HF_HUB_CACHE="/path/to/workspace"
export TRANSFORMERS_CACHE="/path/to/workspace"

# 오프라인 모드 (네트워크 문제 시)
export HF_HUB_OFFLINE=1
```

### 로컬 모델 경로 사용
```yaml
# server_configs/stt_only.yaml
stt:
  model: "/path/to/model/asr_model.nemo"
```

---

## 6. 클라이언트 빌드/실행 오류

### npm SSL 인증서 오류
```bash
# 회사 프록시 환경에서
npm config set strict-ssl false
npm install
```

### Vite 서버 시작 실패
```bash
# node_modules 재설치
rm -rf node_modules package-lock.json
npm install
npm run dev
```

---

## 빠른 복구 스크립트

### restart_server.sh
```bash
#!/bin/bash
# GPU 서버용 서버 재시작 스크립트

echo "Stopping existing servers..."
pkill -f "stt_only_server.py" 2>/dev/null
pkill -f "server.py" 2>/dev/null
sleep 2

echo "Checking ports..."
if lsof -i :7860 > /dev/null 2>&1; then
    echo "Port 7860 still in use, force killing..."
    fuser -k 7860/tcp
fi
if lsof -i :8765 > /dev/null 2>&1; then
    echo "Port 8765 still in use, force killing..."
    fuser -k 8765/tcp
fi
sleep 1

echo "Starting STT-only server..."
cd /path/to/workspace
export NEMO_PATH=/path/to/workspace
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH
export HF_HUB_CACHE="/path/to/workspace"

python server/stt_only_server.py
```

### 사용법
```bash
chmod +x restart_server.sh
./restart_server.sh
```

---

## 환경 설정 요약

### GPU 서버 환경 변수
```bash
# ~/.bashrc 또는 실행 전 설정
export NEMO_PATH=/path/to/workspace
export PYTHONPATH=$NEMO_PATH:$PYTHONPATH
export HF_HUB_CACHE="/path/to/workspace"
export TRANSFORMERS_CACHE="/path/to/workspace"
# export HF_TOKEN="hf_your_token"  # 필요시
```

### 클라이언트 환경 변수 (선택)
```bash
# .env 파일 또는 실행 시
VITE_SERVER_HOST=localhost  # 또는 GPU 서버 IP
```

---

## 문제 발생 시 수집할 정보

이슈 보고 시 다음 정보를 포함:

1. **서버 로그** (시작부터 에러까지)
2. **브라우저 콘솔 로그** (F12 → Console)
3. **네트워크 탭** (F12 → Network → WS 필터)
4. **실행 환경**
   ```bash
   python --version
   node --version
   nvidia-smi
   ```
5. **포트 상태**
   ```bash
   lsof -i :7860
   lsof -i :8765
   ```

---

## 수정된 파일 목록

현재 환경에 맞게 수정된 파일들:

| 파일 | 수정 내용 |
|------|----------|
| `client/vite.config.js` | HTTPS 설정, proxy 설정 |
| `client/src/app.ts` | Premium UI, Protocol auto-detection |
| `client/src/style.css` | Dark theme |
| `client/index.html` | Premium UI layout |
| `nemo/.../websocket_server.py` | Debug logging (선택적) |

---

*Last Updated: 2025-12-12*
