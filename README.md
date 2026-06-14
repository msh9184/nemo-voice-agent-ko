# nemo-voice-agent-ko: Korean Real-Time Voice Agent on NVIDIA NeMo

A real-time, full-duplex **Korean voice agent** built on the [NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo) speech stack and the [pipecat](https://github.com/pipecat-ai/pipecat) orchestration framework. It wires **cache-aware streaming ASR**, **streaming speaker diarization**, **speaker-aware transcript aggregation**, **turn-taking**, **LLM**, and **multi-backend Korean TTS** into one low-latency conversational pipeline with a WebSocket server and a web client.

> Personal research / proof-of-concept project. It extends NVIDIA NeMo's voice-agent example with Korean streaming recognition, multi-speaker handling, and Korean TTS backends. All models referenced are public; model paths in configs are placeholders (`/path/to/...`) for you to fill in.

## Overview

The agent runs as a streaming pipeline: audio frames flow from the client transport through STT (and, optionally, diarization), are aggregated into speaker-attributed text, passed to an LLM, and the response is synthesized by a TTS backend and streamed back — all incrementally, so the user hears partial results with conversational latency.

- **Cache-aware streaming ASR** — NeMo Conformer/RNNT run chunk-by-chunk with carried-over caches for low-latency partial transcripts (Korean + English).
- **Streaming speaker diarization** — `nvidia/diar_streaming_sortformer_4spk-v2` for online up-to-4-speaker labeling.
- **Speaker-aware aggregation** — merges STT + diarization into multi-speaker transcripts (speaker-change breaks, VAD/punctuation boundaries).
- **Turn-taking & backchannel** — endpointing and backchannel detection to manage conversational turns.
- **Pluggable LLM** — any Hugging Face causal LM (Qwen2.5 / Qwen3 / Nemotron / Llama) locally or via a vLLM server.
- **Multi-backend Korean TTS** — Kokoro, MeloTTS-Korean, CosyVoice3, Fish-Speech / OpenAudio.
- **Operating modes** — `stt_only`, `llm_only`, `tts_only`, `stt_llm`, `llm_tts`, `stt_llm_tts`.
- **Runtime-configurable** — change VAD / aggregator / STT / diarization parameters without restarting the server.
- **Web client** — TypeScript/Vite UI with real-time transcript and per-speaker rendering over RTVI.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Real-Time Voice Agent Pipeline                         │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                            │
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
│                                                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
nemo/agents/voice_agent/        # NeMo namespace package (the voice-agent library)
  pipecat/services/nemo/        #   streaming_asr, streaming_diar, stt, tts, llm,
                                #   diar, turn_taking, speaker_aware_aggregator, ...
  pipecat/processors/           #   display + RTVI (speaker-aware) processors
  pipecat/transports/           #   WebSocket server transport
  pipecat/frames/               #   custom diarization/speaker frames
  pipecat/config/ , utils/      #   runtime config + helpers
examples/voice_agent/           # runnable example
  server/                       #   FastAPI + WebSocket server, multi-mode configs
  client/                       #   TypeScript/Vite web client
  scripts/ , *.sh               #   launch / TTS helper scripts
  README.md, SETUP_GUIDE.md, TROUBLESHOOTING.md
patches/nemo-streaming-asr.patch  # core NeMo ASR changes for cache-aware streaming
```

## Streaming-ASR core changes (patch)

Making the NeMo Conformer/RNNT encoder run as a correct cache-aware **streaming** encoder required a small set of changes to NeMo core (mask/cache dimension handling across chunks, plus multilingual language-token support). These are provided as `patches/nemo-streaming-asr.patch` against NVIDIA NeMo and touch:

- `conformer_encoder.py` — chunked forward with `cache_last_channel/time` + trace-friendly mask helpers
- `multi_head_attention.py`, `conformer_modules.py`, `causal_convs.py` — mask/cache dimension fixes for streaming
- `rnnt.py` — language-token support for multilingual models

## Installation

```bash
# 1) Python env with NVIDIA NeMo (ASR/TTS) + pipecat
pip install -r examples/voice_agent/requirements.txt
# (NeMo and pipecat-ai are the core dependencies; see the file for pinned versions)

# 2) Apply the streaming-ASR patch to your NeMo checkout (if using streaming ASR)
#    cd <your-NeMo> && git apply <this-repo>/patches/nemo-streaming-asr.patch

# 3) Web client
cd examples/voice_agent/client && npm install
```

## Quick start

```bash
# Start the server (edit server_configs/*.yaml to point model paths at your models)
cd examples/voice_agent
./run_voice_agent.sh                      # or: python server/server.py

# In another terminal, run the web client
cd client && npm run dev
```

See **[examples/voice_agent/README.md](examples/voice_agent/README.md)** for the full guide, **SETUP_GUIDE.md** for environment setup, and **TROUBLESHOOTING.md** for common issues.

## Models (all public)

| Stage | Example models |
|-------|----------------|
| Streaming ASR | NeMo cache-aware Conformer/RNNT (e.g. `nvidia/parakeet*`) |
| Diarization | `nvidia/diar_streaming_sortformer_4spk-v2` |
| LLM | Qwen2.5 / Qwen3 / Nemotron / Llama (HF or vLLM) |
| TTS | Kokoro-82M, MeloTTS-Korean, CosyVoice3, Fish-Speech / OpenAudio |

Configure your own model paths in `examples/voice_agent/server/server_configs/` — every model path is a `/path/to/...` placeholder by default.

## License & attribution

Licensed under **Apache-2.0** (see [LICENSE](LICENSE)). This project builds on and extends **[NVIDIA NeMo](https://github.com/NVIDIA-NeMo/NeMo)** (Apache-2.0) and uses **[pipecat](https://github.com/pipecat-ai/pipecat)** for pipeline orchestration. Model and TTS backends are the property of their respective authors.

## Acknowledgements

NVIDIA NeMo (ASR, Sortformer diarization, TTS), the pipecat framework, and the open-source TTS/LLM model authors.
