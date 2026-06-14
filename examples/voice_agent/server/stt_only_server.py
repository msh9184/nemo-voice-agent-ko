# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
STT-Only Server Entry Point

This server provides speech-to-text functionality without LLM or TTS.
Use this for standalone ASR services.

Usage:
    export SERVER_CONFIG_PATH="./server/server_configs/stt_only_mode.yaml"
    python ./server/stt_only_server.py

The server will:
1. Start FastAPI on port 7860 for /connect endpoint
2. Start WebSocket server on port 8765 for audio streaming and transcription

Client can connect using the standard voice_agent client or any WebSocket client
that implements the RTVI protocol.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(override=True)

# Set default config path to stt_only_mode.yaml if not specified
if not os.environ.get("SERVER_CONFIG_PATH"):
    os.environ["SERVER_CONFIG_PATH"] = os.path.join(
        os.path.dirname(__file__), "server_configs", "stt_only_mode.yaml"
    )

from stt_only_bot_websocket_server import run_stt_only_websocket_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    print("=" * 60)
    print("STT-Only Server Starting...")
    print("=" * 60)
    yield
    print("STT-Only Server Shutting Down...")


app = FastAPI(
    title="NeMo STT-Only Server",
    description="Speech-to-Text only server without LLM and TTS",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "mode": "stt_only",
        "description": "NeMo STT-Only Server - Speech recognition without LLM/TTS"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "mode": "stt_only"}


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    """
    Return WebSocket URL for client connection.
    This endpoint is compatible with the standard voice_agent client.

    Auto-detects protocol (ws/wss) based on request scheme to prevent
    Mixed Content errors when client is loaded via HTTPS.
    """
    print("Received /connect request for STT-only mode")
    server_host = request.url.hostname or request.headers.get("host", "").split(":")[0]

    # Check if request came via HTTPS (directly or through proxy)
    # X-Forwarded-Proto is set by reverse proxies
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    is_secure = request.url.scheme == "https" or forwarded_proto == "https"

    ws_protocol = "wss" if is_secure else "ws"
    ws_url = f"{ws_protocol}://{server_host}:8765"
    print(f"Returning WebSocket URL: {ws_url} (secure={is_secure})")
    return {"ws_url": ws_url}


async def main():
    """Main entry point - runs both FastAPI and WebSocket servers"""
    print("\n" + "=" * 60)
    print("NeMo STT-Only Server")
    print("=" * 60)
    print(f"Config: {os.environ.get('SERVER_CONFIG_PATH')}")
    print("FastAPI: http://0.0.0.0:7860")
    print("WebSocket: ws://0.0.0.0:8765")
    print("=" * 60 + "\n")

    tasks = []
    try:
        # Start STT-only WebSocket server
        tasks.append(run_stt_only_websocket_server())

        # Start FastAPI server
        config = uvicorn.Config(app, host="0.0.0.0", port=7860, log_level="info")
        server = uvicorn.Server(config)
        tasks.append(server.serve())

        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Tasks cancelled (shutdown)")
    except KeyboardInterrupt:
        print("Keyboard interrupt received")


if __name__ == "__main__":
    asyncio.run(main())
