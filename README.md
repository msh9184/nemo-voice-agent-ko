# 🎙️ nemo-voice-agent-ko

**Real-time, full-duplex Korean voice agent built on NVIDIA NeMo + pipecat.**
Streaming ASR · streaming speaker diarization · speaker-aware aggregation · turn-taking · LLM · multi-backend Korean TTS — wired into one low-latency conversational pipeline with a WebSocket server and a web client.

<p>
<img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue">
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
<img alt="built on" src="https://img.shields.io/badge/built%20on-NVIDIA%20NeMo-76B900">
<img alt="framework" src="https://img.shields.io/badge/pipeline-pipecat-555">
<img alt="status" src="https://img.shields.io/badge/status-research%20PoC-orange">
</p>

> Personal research / proof-of-concept extending **[NVIDIA NeMo's voice-agent example](https://github.com/NVIDIA-NeMo/NeMo)** with Korean streaming recognition, multi-speaker handling, and Korean TTS backends. Every model referenced is public; model paths in configs are `/path/to/...` placeholders for you to fill in.

---

## What is this?

A microphone-to-speaker conversational agent for **Korean** that streams the whole pipeline so the user hears partial results at conversational latency:

```
audio ─▶ streaming ASR ─▶ (streaming diarization ─▶ speaker-aware aggregation) ─▶ turn-taking ─▶ LLM ─▶ TTS ─▶ audio
```

It runs as a FastAPI WebSocket **server** plus a TypeScript/Vite **web client**, and can operate in six independently selectable modes (STT-only ↔ full duplex).

## ✨ Key features

- **Cache-aware streaming ASR (Korean + multilingual)** — NeMo Conformer/RNNT run chunk-by-chunk with carried-over caches for low-latency partial transcripts. Enabled by a 5-file NeMo-core patch (see [`patches/`](patches/)).
- **Streaming speaker diarization** — online `nvidia/diar_streaming_sortformer_4spk-v2`, up to 4 speakers.
- **Speaker-aware aggregation** — merges STT + diarization into multi-speaker transcripts (speaker-change / VAD / punctuation boundaries, color-coded console).
- **Turn-taking & backchannel** — Korean backchannel detection and endpointing for natural barge-in.
- **Pluggable LLM** — any Hugging Face causal LM (Qwen2.5 / Qwen3 / Nemotron / Llama) locally or via a vLLM server.
- **Multi-backend Korean TTS** — Kokoro, MeloTTS-Korean, CosyVoice3, Fish-Speech / OpenAudio (incl. zero-shot voice cloning).
- **Six operating modes** — `stt_only`, `llm_only`, `tts_only`, `stt_llm`, `llm_tts`, `stt_llm_tts`, all from one server.
- **Runtime-reconfigurable** — change VAD / aggregator / STT / diarization params live, no restart.
- **Web client with per-speaker UI** — real-time transcript, speaker colors, module/model badges, runtime config panel.

## 🆚 How this differs from upstream NVIDIA NeMo's voice-agent example

| Capability | Upstream example | **This repo** |
|---|---|---|
| Cache-aware **streaming** Conformer/RNNT | — | ✅ 5-file NeMo-core patch (mask/cache dims + multilingual language tokens) |
| Streaming **speaker diarization** | — | ✅ Sortformer, ≤4 speakers, online |
| Speaker-aware transcript aggregation | — | ✅ multi-speaker, boundary-aware |
| Korean TTS backends | — | ✅ Kokoro / MeloTTS-KO / CosyVoice3 / Fish-Speech |
| Operating modes | single | ✅ 6 selectable modes |
| Runtime reconfiguration | — | ✅ live VAD/aggregator/STT/diar |
| Web client speaker UI | basic | ✅ per-speaker colors + module/model badges |
| Korean-first prompting / backchannel | — | ✅ Korean personas, `<lan_xx>`/special-token handling |

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Real-Time Voice Agent Pipeline                         │
├──────────────────────────────────────────────────────────────────────────┤
│  Mic / audio (16 kHz)                                                       │
│        │   WebSocket (pipecat transport)                                    │
│        ▼                                                                    │
│  ┌──────────────┐   ┌─────────────────────┐                                │
│  │ Streaming ASR │──▶│ Streaming Diar      │  (Sortformer, ≤4 spk)          │
│  │ (Conformer/   │   │ + Speaker-Aware     │                                │
│  │  RNNT, cache) │   │   Aggregator        │                                │
│  └──────────────┘   └──────────┬──────────┘                                │
│        │ partial/final text                │ speaker-attributed text       │
│        └──────────────┬─────────────────────┘                              │
│                       ▼                                                     │
│                 ┌────────────┐   turn-taking / endpointing                  │
│                 │    LLM     │   (HF causal LM or vLLM)                      │
│                 └─────┬──────┘                                              │
│                       ▼                                                     │
│                 ┌────────────┐                                             │
│                 │    TTS     │  (Kokoro / MeloTTS-KO / CosyVoice3 / Fish)   │
│                 └─────┬──────┘                                             │
│                       ▼                                                     │
│             streamed audio + transcript ──▶ Web client (RTVI)              │
└──────────────────────────────────────────────────────────────────────────┘
```

## 📁 Repository layout

```
nemo/agents/voice_agent/          # NeMo namespace package (the voice-agent library)
  pipecat/services/nemo/          #   streaming_asr, streaming_diar, stt, tts, llm,
                                  #   diar, turn_taking, speaker_aware_aggregator, ...
  pipecat/processors/             #   display + RTVI (speaker-aware) processors
  pipecat/transports/             #   WebSocket server transport
  pipecat/frames/ , config/ , utils/
examples/voice_agent/             # runnable example
  server/                         #   FastAPI + WebSocket server, 6-mode configs
  client/                         #   TypeScript/Vite web client
  scripts/ , *.sh                 #   launch / TTS helper scripts
  README.md, SETUP_GUIDE.md, TROUBLESHOOTING.md   # deep-dive docs
patches/nemo-streaming-asr.patch  # NeMo-core changes for cache-aware streaming ASR
```

## 🚀 Quick start

```bash
# 1) Install (pip path)
pip install -r examples/voice_agent/requirements.txt
cd examples/voice_agent/client && npm install && cd -

# 2) Apply the streaming-ASR patch to your NeMo checkout
cd <your-NeMo> && git apply <this-repo>/patches/nemo-streaming-asr.patch && cd -

# 3) Easiest first run — STT-only (no LLM/TTS needed)
cd examples/voice_agent && ./run_voice_agent.sh -m stt-only

# 4) In another terminal, start the web client → open http://localhost:5173
cd examples/voice_agent/client && npm run dev
```

Then scale up to the full agent (set the `/path/to/...` model paths in `server/server_configs/*.yaml` first):

```bash
cd examples/voice_agent && ./run_voice_agent.sh -m full      # self-contained (FastPitch/Kokoro TTS)
./stop_voice_agent.sh
```

> Conda alternative: `conda env create -f examples/voice_agent/environment.yaml && conda activate nemo-voice`.
> `requirements.txt` is the canonical dependency set.

## 🎛️ Operating modes

| Mode | Config | STT | LLM | TTS | In → Out |
|---|---|:--:|:--:|:--:|---|
| Full | `stt_llm_tts_mode.yaml` | ✅ | ✅ | ✅ | voice → voice |
| STT + LLM | `stt_llm_mode.yaml` | ✅ | ✅ | — | voice → text |
| STT-only | `stt_only_mode.yaml` | ✅ | — | — | voice → text |
| LLM + TTS | `llm_tts_mode.yaml` | — | ✅ | ✅ | text → voice |
| LLM-only | `llm_only_mode.yaml` | — | ✅ | — | text → text |
| TTS-only | `tts_only_mode.yaml` | — | — | ✅ | text → voice |

`./run_voice_agent.sh -m {stt-only|stt-llm|full|llm-tts|llm-only|tts-only}` selects a mode. See **[examples/voice_agent/README.md](examples/voice_agent/README.md)** for the full configuration reference.

## 🧪 Example output

Server boot (STT-only):

```
INFO | Loading NeMo cache-aware streaming ASR ...
INFO | STT service initialized (Conformer/RNNT, chunk streaming)
INFO | Starting websocket server on 0.0.0.0:8765
```

CLI audio-file client (`python examples/voice_agent/audio_file_client.py --file sample.wav`):

```
┌──────────────────── Audio File Client ────────────────────┐
[001] 안녕하세요, 오늘 회의를 시작하겠습니다.          [|]
[002] 네, 자료 먼저 공유드릴게요.                        [.]
─────────────────────────────────────────────────────────────
SESSION SUMMARY  ·  utterances: 2  ·  RTF: 0.18
```

CLI text client (`python examples/voice_agent/text_client.py`):

```
  [You]        안녕하세요, 오늘 날씨가 어때요?
  [Assistant]  안녕하세요! 저는 AI 어시스턴트라 실시간 날씨를 직접 확인할 수는 없어요.
               지역을 알려주시면 이어서 도와드릴게요.
```

## 🩹 The streaming-ASR core patch

`patches/nemo-streaming-asr.patch` turns the NeMo Conformer/RNNT encoder into a correct **cache-aware streaming** encoder and adds multilingual language-token support. It touches 5 core files (`conformer_encoder.py`, `multi_head_attention.py`, `conformer_modules.py`, `causal_convs.py`, `rnnt.py`). See [`patches/README.md`](patches/README.md) for details and how to apply.

## ✅ Tests

```bash
pytest examples/voice_agent/tests/test_config_manager.py
```

## 📜 License & attribution

Apache-2.0 (see [LICENSE](LICENSE)). Builds on and extends **[NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo)** (Apache-2.0) and uses **[pipecat](https://github.com/pipecat-ai/pipecat)** for pipeline orchestration. ASR/diarization/TTS/LLM models are property of their respective authors.

## 🙏 Acknowledgements

NVIDIA NeMo (ASR, Sortformer diarization, TTS), the pipecat framework, and the open-source Korean TTS / LLM model authors.
