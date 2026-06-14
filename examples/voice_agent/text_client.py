#!/usr/bin/env python3
"""
Premium CLI Text Client for Voice Agent Text Input Modes

A professional-grade command line client for interacting with Voice Agent server
in text input modes (LLM-Only, TTS-Only, LLM+TTS).

Features:
- Premium console UI with visual indicators
- Real-time streaming text output
- Intelligent text normalization (removes excessive whitespace)
- Server configuration auto-detection
- Rich status display

Usage:
    python text_client.py [--host HOST] [--port PORT]

Examples:
    python text_client.py                     # Connect to localhost:8765
    python text_client.py --host 10.0.0.1     # Connect to specific host
"""

import asyncio
import json
import uuid
import argparse
import sys
import re
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import readline  # For better command line editing (Unix)
except ImportError:
    pass  # Windows doesn't have readline

try:
    import websockets
except ImportError:
    print("Error: websockets package not installed.")
    print("Install with: pip install websockets")
    sys.exit(1)


class TextClient:
    """Premium CLI text client for Voice Agent server."""

    # ═══════════════════════════════════════════════════════════════════
    # ANSI Escape Codes
    # ═══════════════════════════════════════════════════════════════════

    # Style codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"

    # Foreground colors (standard)
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Foreground colors (bright)
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Alias for convenience
    GRAY = BRIGHT_BLACK

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Special Unicode characters for premium UI
    CHAR_BOX_TL = "╔"
    CHAR_BOX_TR = "╗"
    CHAR_BOX_BL = "╚"
    CHAR_BOX_BR = "╝"
    CHAR_BOX_H = "═"
    CHAR_BOX_V = "║"
    CHAR_LINE = "─"
    CHAR_ARROW_RIGHT = "▶"
    CHAR_ARROW_DOWN = "▼"
    CHAR_DOT = "●"
    CHAR_CIRCLE = "○"
    CHAR_CHECK = "✓"
    CHAR_CROSS = "✗"
    CHAR_STAR = "★"
    CHAR_DIAMOND = "◆"
    CHAR_TRIANGLE = "▲"
    CHAR_SQUARE = "■"
    CHAR_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.ws_url = f"ws://{host}:{port}"
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.running = True

        # Server configuration
        self.service_mode = "unknown"
        self.input_mode = "unknown"
        self.llm_enabled = False
        self.tts_enabled = False
        self.stt_enabled = False
        self.llm_model = "unknown"
        self.tts_model = "unknown"
        self.stt_model = "unknown"
        self.config_received = False

        # Response state
        self.current_response = ""
        self.response_buffer = ""  # Buffer for text normalization
        self.llm_streaming = False
        self.response_complete = False
        self.message_count = 0
        self.last_response_text = ""

    # ═══════════════════════════════════════════════════════════════════
    # Utility Methods
    # ═══════════════════════════════════════════════════════════════════

    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%H:%M:%S")

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing excessive whitespace and fixing formatting.

        This handles the issue where LLM/TTS outputs have excessive spaces
        between words due to token-by-token or chunk-by-chunk streaming.
        """
        if not text:
            return text

        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)

        # Fix space before punctuation
        text = re.sub(r'\s+([.,!?;:)])', r'\1', text)

        # Fix space after opening brackets
        text = re.sub(r'([(])\s+', r'\1', text)

        # Normalize line breaks (multiple newlines to double newline)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        return text.strip()

    def _wrap_text(self, text: str, width: int = 70, indent: str = "  ") -> str:
        """Wrap text to specified width with indentation."""
        words = text.split()
        lines = []
        current_line = indent

        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                if current_line == indent:
                    current_line += word
                else:
                    current_line += " " + word
            else:
                if current_line != indent:
                    lines.append(current_line)
                current_line = indent + word

        if current_line != indent:
            lines.append(current_line)

        return '\n'.join(lines) if lines else indent

    # ═══════════════════════════════════════════════════════════════════
    # UI Components
    # ═══════════════════════════════════════════════════════════════════

    def _print_header(self):
        """Print premium application header."""
        print()
        print(f"{self.BRIGHT_CYAN}{self.CHAR_BOX_TL}{self.CHAR_BOX_H * 62}{self.CHAR_BOX_TR}{self.RESET}")
        print(f"{self.BRIGHT_CYAN}{self.CHAR_BOX_V}{self.RESET}  {self.BOLD}{self.BRIGHT_WHITE}{self.CHAR_DIAMOND} Voice Agent Text Client{self.RESET}                                  {self.BRIGHT_CYAN}{self.CHAR_BOX_V}{self.RESET}")
        print(f"{self.BRIGHT_CYAN}{self.CHAR_BOX_V}{self.RESET}  {self.DIM}Premium CLI for LLM-Only / TTS-Only / LLM+TTS Modes{self.RESET}           {self.BRIGHT_CYAN}{self.CHAR_BOX_V}{self.RESET}")
        print(f"{self.BRIGHT_CYAN}{self.CHAR_BOX_V}{self.CHAR_BOX_H * 62}{self.CHAR_BOX_V}{self.RESET}")
        print(f"{self.BRIGHT_CYAN}{self.CHAR_BOX_V}{self.RESET}  {self.DIM}Commands:{self.RESET} {self.CYAN}help{self.RESET} {self.DIM}|{self.RESET} {self.CYAN}status{self.RESET} {self.DIM}|{self.RESET} {self.CYAN}config{self.RESET} {self.DIM}|{self.RESET} {self.CYAN}clear{self.RESET} {self.DIM}|{self.RESET} {self.CYAN}quit{self.RESET}               {self.BRIGHT_CYAN}{self.CHAR_BOX_V}{self.RESET}")
        print(f"{self.BRIGHT_CYAN}{self.CHAR_BOX_BL}{self.CHAR_BOX_H * 62}{self.CHAR_BOX_BR}{self.RESET}")
        print()

    def _print_box(self, title: str, content_lines: list, color: str = None):
        """Print a styled box with title and content."""
        if color is None:
            color = self.BRIGHT_CYAN

        max_len = max(len(title), max(len(line) for line in content_lines) if content_lines else 0)
        width = max(max_len + 4, 50)

        print()
        print(f"{color}{self.CHAR_BOX_TL}{self.CHAR_BOX_H * width}{self.CHAR_BOX_TR}{self.RESET}")
        print(f"{color}{self.CHAR_BOX_V}{self.RESET}  {self.BOLD}{title}{self.RESET}{' ' * (width - len(title) - 2)}{color}{self.CHAR_BOX_V}{self.RESET}")
        print(f"{color}{self.CHAR_BOX_V}{self.CHAR_LINE * width}{self.CHAR_BOX_V}{self.RESET}")

        for line in content_lines:
            padding = width - len(self._strip_ansi(line)) - 2
            print(f"{color}{self.CHAR_BOX_V}{self.RESET}  {line}{' ' * padding}{color}{self.CHAR_BOX_V}{self.RESET}")

        print(f"{color}{self.CHAR_BOX_BL}{self.CHAR_BOX_H * width}{self.CHAR_BOX_BR}{self.RESET}")
        print()

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text for length calculation."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def _get_prompt(self) -> str:
        """Get the input prompt string."""
        if self.llm_streaming:
            return ""  # Don't show prompt during streaming

        # Premium prompt with visual indicator
        indicator = f"{self.BRIGHT_GREEN}{self.CHAR_DOT}{self.RESET}" if self.connected else f"{self.RED}{self.CHAR_CIRCLE}{self.RESET}"
        mode_indicator = ""

        if self.config_received:
            if self.service_mode == "llm_only":
                mode_indicator = f"{self.CYAN}LLM{self.RESET}"
            elif self.service_mode == "tts_only":
                mode_indicator = f"{self.MAGENTA}TTS{self.RESET}"
            elif self.service_mode == "llm_tts":
                mode_indicator = f"{self.GREEN}LLM+TTS{self.RESET}"
            else:
                mode_indicator = f"{self.YELLOW}{self.service_mode}{self.RESET}"

        if mode_indicator:
            return f"\n{indicator} [{mode_indicator}] {self.BRIGHT_WHITE}{self.CHAR_ARROW_RIGHT}{self.RESET} "
        else:
            return f"\n{indicator} {self.BRIGHT_WHITE}{self.CHAR_ARROW_RIGHT}{self.RESET} "

    # ═══════════════════════════════════════════════════════════════════
    # Logging & Output
    # ═══════════════════════════════════════════════════════════════════

    def log(self, level: str, message: str):
        """Print timestamped log message with premium formatting."""
        timestamp = self._get_timestamp()

        formats = {
            "info": (self.CYAN, f"{self.CHAR_TRIANGLE}", "INFO"),
            "success": (self.BRIGHT_GREEN, self.CHAR_CHECK, "OK"),
            "warning": (self.YELLOW, "!", "WARN"),
            "error": (self.BRIGHT_RED, self.CHAR_CROSS, "ERR"),
            "system": (self.MAGENTA, self.CHAR_STAR, "SYS"),
            "debug": (self.GRAY, self.CHAR_CIRCLE, "DBG"),
        }

        color, icon, label = formats.get(level, (self.WHITE, self.CHAR_DOT, level.upper()))
        print(f"{self.GRAY}{timestamp}{self.RESET} {color}[{icon}]{self.RESET} {message}")

    def _print_user_message(self, text: str):
        """Print user message with premium styling."""
        self.message_count += 1
        timestamp = self._get_timestamp()

        print()
        print(f"  {self.GRAY}{timestamp}{self.RESET} {self.BRIGHT_BLUE}{self.BOLD}[You]{self.RESET} {self.DIM}#{self.message_count}{self.RESET}")
        print(f"  {self.GRAY}{self.CHAR_LINE * 40}{self.RESET}")

        # Wrap and indent message text
        wrapped = self._wrap_text(text, width=60, indent="  ")
        print(f"{self.WHITE}{wrapped}{self.RESET}")
        print()

    def _start_assistant_response(self):
        """Start assistant response block."""
        if self.llm_streaming:
            return  # Already streaming

        timestamp = self._get_timestamp()
        print()
        print(f"  {self.GRAY}{timestamp}{self.RESET} {self.BRIGHT_GREEN}{self.BOLD}[Assistant]{self.RESET}")
        print(f"  {self.GRAY}{self.CHAR_LINE * 40}{self.RESET}")
        print(f"  {self.WHITE}", end="", flush=True)
        self.llm_streaming = True
        self.response_buffer = ""

    def _stream_assistant_text(self, text: str):
        """Stream assistant text with intelligent buffering and normalization."""
        if not self.llm_streaming:
            self._start_assistant_response()

        # Add to buffer
        self.response_buffer += text

        # Normalize and print periodically (when we have sentence boundaries or enough text)
        if any(c in text for c in '.!?\n') or len(self.response_buffer) > 100:
            normalized = self._normalize_text(self.response_buffer)

            # Only print if we have new content after the last printed text
            if normalized and normalized != self.last_response_text:
                # Calculate what's new
                if self.last_response_text and normalized.startswith(self.last_response_text):
                    new_text = normalized[len(self.last_response_text):]
                else:
                    new_text = normalized

                if new_text.strip():
                    # Handle newlines properly
                    if '\n' in new_text:
                        lines = new_text.split('\n')
                        for i, line in enumerate(lines):
                            if i > 0:
                                print()  # New line
                                print(f"  ", end="")  # Indent
                            print(line, end="", flush=True)
                    else:
                        print(new_text, end="", flush=True)

                    self.last_response_text = normalized

    def _end_assistant_response(self):
        """End assistant response block with final normalization."""
        if self.llm_streaming:
            # Print any remaining buffered text
            if self.response_buffer:
                normalized = self._normalize_text(self.response_buffer)
                if normalized and normalized != self.last_response_text:
                    remaining = normalized[len(self.last_response_text):] if self.last_response_text else normalized
                    if remaining.strip():
                        print(remaining, end="")

            print(f"{self.RESET}")  # End line and reset
            print()
            self.llm_streaming = False
            self.response_buffer = ""
            self.last_response_text = ""

    def _print_tts_status(self, text: str):
        """Print TTS synthesis status (visual indicator only)."""
        if not self.tts_enabled:
            return

        timestamp = self._get_timestamp()
        display_text = text[:50] + "..." if len(text) > 50 else text
        display_text = self._normalize_text(display_text)
        print(f"\r  {self.GRAY}{timestamp}{self.RESET} {self.MAGENTA}[TTS {self.CHAR_TRIANGLE}]{self.RESET} {self.DIM}{display_text}{self.RESET}", end="", flush=True)

    # ═══════════════════════════════════════════════════════════════════
    # Status & Help Display
    # ═══════════════════════════════════════════════════════════════════

    def print_status(self):
        """Print detailed connection and service status."""
        connected_str = f"{self.BRIGHT_GREEN}{self.CHAR_CHECK} Connected{self.RESET}" if self.connected else f"{self.BRIGHT_RED}{self.CHAR_CROSS} Disconnected{self.RESET}"

        # Mode display
        mode_colors = {
            "llm_only": self.CYAN,
            "tts_only": self.MAGENTA,
            "llm_tts": self.GREEN,
            "full": self.YELLOW,
            "stt_llm": self.BLUE,
            "stt_only": self.CYAN,
        }
        mode_color = mode_colors.get(self.service_mode, self.WHITE)
        mode_str = f"{mode_color}{self.service_mode}{self.RESET}"

        # Component status
        def status_str(enabled: bool, name: str, model: str = ""):
            if enabled:
                model_info = f" ({model})" if model and model != "unknown" else ""
                return f"{self.BRIGHT_GREEN}{self.CHAR_CHECK} {name}{self.RESET}{self.DIM}{model_info}{self.RESET}"
            else:
                return f"{self.GRAY}{self.CHAR_CIRCLE} {name}{self.RESET}"

        content = [
            f"Server: {self.CYAN}{self.ws_url}{self.RESET}",
            f"Status: {connected_str}",
            f"Mode: {mode_str}",
            f"Input: {self.input_mode}",
            "",
            f"{status_str(self.stt_enabled, 'STT', self.stt_model)}",
            f"{status_str(self.llm_enabled, 'LLM', self.llm_model)}",
            f"{status_str(self.tts_enabled, 'TTS', self.tts_model)}",
        ]

        self._print_box("Connection Status", content)

    def print_config(self):
        """Print server configuration details."""
        if not self.config_received:
            self.log("warning", "No configuration received from server yet")
            self.log("info", "Requesting configuration...")
            return False

        content = [
            f"Service Mode: {self.BOLD}{self.service_mode}{self.RESET}",
            f"Input Mode: {self.input_mode}",
            "",
            f"{self.UNDERLINE}Components:{self.RESET}",
            f"  STT: {'Enabled' if self.stt_enabled else 'Disabled'}",
            f"  LLM: {'Enabled' if self.llm_enabled else 'Disabled'}",
            f"  TTS: {'Enabled' if self.tts_enabled else 'Disabled'}",
        ]

        if self.llm_model != "unknown":
            content.append(f"  LLM Model: {self.llm_model}")
        if self.tts_model != "unknown":
            content.append(f"  TTS Model: {self.tts_model}")

        self._print_box("Server Configuration", content, self.BRIGHT_MAGENTA)
        return True

    def print_help(self):
        """Print help information."""
        content = [
            f"{self.CYAN}<text>{self.RESET}     Send text message to server",
            "",
            f"{self.CYAN}status{self.RESET}    Show connection status",
            f"{self.CYAN}config{self.RESET}    Show server configuration",
            f"{self.CYAN}clear{self.RESET}     Clear conversation history",
            f"{self.CYAN}help{self.RESET}      Show this help message",
            f"{self.CYAN}quit{self.RESET}      Exit the client",
            "",
            f"{self.DIM}Shortcuts: q=quit, ?=help, Ctrl+C=exit{self.RESET}",
        ]

        self._print_box("Available Commands", content, self.BRIGHT_YELLOW)

    def print_mode_banner(self):
        """Print current mode banner after configuration is received."""
        mode_info = {
            "llm_only": ("LLM-Only Mode", "Text → LLM → Text", self.CYAN),
            "tts_only": ("TTS-Only Mode", "Text → TTS → Audio", self.MAGENTA),
            "llm_tts": ("LLM+TTS Mode", "Text → LLM → TTS → Audio", self.GREEN),
            "full": ("Full Voice Mode", "Voice → STT → LLM → TTS → Audio", self.YELLOW),
            "stt_llm": ("STT+LLM Mode", "Voice → STT → LLM → Text", self.BLUE),
            "stt_only": ("STT-Only Mode", "Voice → STT → Text", self.CYAN),
        }

        mode_name, flow, color = mode_info.get(
            self.service_mode,
            (self.service_mode, "Unknown", self.WHITE)
        )

        print()
        print(f"  {color}{self.CHAR_BOX_H * 50}{self.RESET}")
        print(f"  {color}{self.CHAR_STAR}{self.RESET} {self.BOLD}{mode_name}{self.RESET}")
        print(f"  {self.DIM}Flow: {flow}{self.RESET}")
        print(f"  {color}{self.CHAR_BOX_H * 50}{self.RESET}")
        print()

        if self.input_mode == "text":
            print(f"  {self.BRIGHT_GREEN}{self.CHAR_CHECK} Ready for input{self.RESET}")
            print(f"  {self.DIM}Type your message and press Enter to send{self.RESET}")
        else:
            self.log("warning", "Server is in voice input mode - text input may not work as expected")

    # ═══════════════════════════════════════════════════════════════════
    # WebSocket Communication
    # ═══════════════════════════════════════════════════════════════════

    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        self.log("info", f"Connecting to {self.CYAN}{self.ws_url}{self.RESET}...")

        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
                max_size=10 * 1024 * 1024  # 10MB max message size
            )
            self.connected = True
            self.log("success", "WebSocket connection established")

            # Send client-ready and request config
            await self._send_client_ready()
            await self._request_server_config()

            return True

        except ConnectionRefusedError:
            self.log("error", "Connection refused - is the server running?")
            return False
        except Exception as e:
            self.log("error", f"Failed to connect: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        self.running = False
        if self.websocket:
            try:
                await self.websocket.close()
                self.log("info", "Disconnected from server")
            except Exception as e:
                self.log("warning", f"Error during disconnect: {e}")
        self.connected = False

    async def _send_message(self, message: dict):
        """Send a JSON message to the server."""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                self.log("error", f"Failed to send message: {e}")

    async def _send_client_ready(self):
        """Send RTVI client-ready message."""
        message = {
            "id": str(uuid.uuid4()),
            "type": "client-ready",
            "data": {
                "config": {
                    "audio_in_enabled": False,
                    "audio_out_enabled": True
                }
            }
        }
        await self._send_message(message)

    async def _request_server_config(self):
        """Request server configuration via RTVI action."""
        message = {
            "id": str(uuid.uuid4()),
            "type": "action",
            "data": {
                "service": "config",
                "action": "get_server_config",
                "arguments": []
            }
        }
        await self._send_message(message)

    async def send_text(self, text: str):
        """Send text input via RTVI action."""
        if not self.connected or not self.websocket:
            self.log("error", "Not connected to server")
            return

        # Reset response state
        self.current_response = ""
        self.response_buffer = ""
        self.last_response_text = ""
        self.response_complete = False
        self.llm_streaming = False

        # Create RTVI action message
        message = {
            "id": str(uuid.uuid4()),
            "type": "action",
            "data": {
                "service": "text_input",
                "action": "send",
                "arguments": [
                    {"name": "text", "value": text}
                ]
            }
        }

        self._print_user_message(text)
        await self._send_message(message)

    # ═══════════════════════════════════════════════════════════════════
    # Message Processing
    # ═══════════════════════════════════════════════════════════════════

    def _extract_json_from_protobuf(self, data: bytes) -> list:
        """Extract JSON messages from Protobuf-framed binary data."""
        results = []

        # Try direct UTF-8 decode
        try:
            text = data.decode('utf-8')
            try:
                results.append(json.loads(text))
                return results
            except json.JSONDecodeError:
                pass
        except UnicodeDecodeError:
            pass

        # Try to find JSON objects in binary data
        try:
            data_str = data.decode('latin-1')
        except:
            return results

        # Find JSON objects by matching braces
        i = 0
        while i < len(data_str):
            if data_str[i] == '{':
                brace_count = 1
                j = i + 1
                while j < len(data_str) and brace_count > 0:
                    if data_str[j] == '{':
                        brace_count += 1
                    elif data_str[j] == '}':
                        brace_count -= 1
                    j += 1

                if brace_count == 0:
                    try:
                        json_str = data_str[i:j]
                        obj = json.loads(json_str)
                        results.append(obj)
                        i = j
                        continue
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            i += 1

        return results

    async def handle_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    if isinstance(message, str):
                        data = json.loads(message)
                        await self._process_message(data)
                    elif isinstance(message, bytes):
                        json_objects = self._extract_json_from_protobuf(message)
                        for data in json_objects:
                            await self._process_message(data)

                except json.JSONDecodeError:
                    pass  # Ignore invalid JSON
                except Exception as e:
                    self.log("error", f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            self.log("warning", f"Connection closed: {e}")
            self.connected = False
        except Exception as e:
            self.log("error", f"Error in message handler: {e}")
            self.connected = False

    async def _process_message(self, data: dict):
        """Process received RTVI message."""
        msg_type = data.get("type", "")
        msg_data = data.get("data", {})

        if msg_type == "bot-ready":
            self.log("success", "Bot is ready")

        elif msg_type == "server-config":
            self._handle_server_config(msg_data)

        elif msg_type == "bot-llm-text":
            # LLM text chunk (streaming)
            text = msg_data.get("text", "")
            if text:
                self.current_response += text
                self._stream_assistant_text(text)

        elif msg_type == "bot-llm-started":
            self._start_assistant_response()

        elif msg_type in ("bot-llm-stopped", "bot-llm-text-aggregation"):
            self._end_assistant_response()
            self.response_complete = True

        elif msg_type == "bot-tts-text":
            text = msg_data.get("text", "")
            if text:
                self._print_tts_status(text)

        elif msg_type == "bot-tts-started":
            pass  # TTS started

        elif msg_type == "bot-tts-stopped":
            if self.tts_enabled:
                print()  # Clear TTS status line

        elif msg_type == "bot-status":
            status = msg_data.get("status", "")
            if status == "idle" and self.llm_streaming:
                self._end_assistant_response()

        elif msg_type == "action-response":
            result = msg_data.get("result", {})
            if not result.get("success", True):
                error = result.get("error", "Unknown error")
                self.log("error", f"Action failed: {error}")

        elif msg_type == "error":
            error = msg_data.get("message", "Unknown error")
            self.log("error", f"Server error: {error}")

    def _handle_server_config(self, config: dict):
        """Process server configuration."""
        self.service_mode = config.get("service_mode", "unknown")
        self.input_mode = config.get("input_mode", "unknown")

        # Parse component configurations
        stt_config = config.get("stt", {})
        llm_config = config.get("llm", {})
        tts_config = config.get("tts", {})

        self.stt_enabled = stt_config.get("enabled", False)
        self.llm_enabled = llm_config.get("enabled", False)
        self.tts_enabled = tts_config.get("enabled", False)

        # Get model names if available
        self.stt_model = stt_config.get("model", "unknown")
        self.llm_model = llm_config.get("model", "unknown")
        self.tts_model = tts_config.get("model", "unknown")

        self.config_received = True

        # Display mode banner
        self.print_mode_banner()


async def input_loop(client: TextClient):
    """Handle user input in a separate task."""
    loop = asyncio.get_event_loop()

    # Wait a moment for config to arrive
    await asyncio.sleep(0.5)

    while client.running and client.connected:
        try:
            # Get input with premium prompt
            prompt = client._get_prompt()
            user_input = await loop.run_in_executor(
                None,
                lambda: input(prompt).strip()
            )

            if not user_input:
                continue

            # Handle commands
            cmd = user_input.lower()

            if cmd in ("quit", "exit", "q"):
                print()
                client.log("info", "Goodbye!")
                await client.disconnect()
                break

            elif cmd == "status":
                client.print_status()

            elif cmd == "config":
                if not client.print_config():
                    await client._request_server_config()

            elif cmd in ("help", "?"):
                client.print_help()

            elif cmd == "clear":
                message = {
                    "id": str(uuid.uuid4()),
                    "type": "action",
                    "data": {
                        "service": "context",
                        "action": "reset",
                        "arguments": []
                    }
                }
                await client._send_message(message)
                client.message_count = 0
                client.log("success", "Conversation cleared")

            else:
                # Send text input
                await client.send_text(user_input)

        except EOFError:
            print()
            client.log("info", "EOF received, disconnecting...")
            await client.disconnect()
            break
        except KeyboardInterrupt:
            print()
            client.log("info", "Interrupted, disconnecting...")
            await client.disconnect()
            break
        except Exception as e:
            client.log("error", f"Input error: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Premium CLI Text Client for Voice Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python text_client.py                     # Connect to localhost:8765
    python text_client.py --host 10.0.0.1     # Connect to specific host
    python text_client.py --port 8080         # Connect to specific port

Server Configuration:
    Make sure the server is running with a text input mode config:

    # LLM-Only mode (Text -> LLM -> Text)
    export SERVER_CONFIG_PATH=./server/server_configs/llm_only_mode.yaml
    python ./server/server.py
        """
    )
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    args = parser.parse_args()

    client = TextClient(host=args.host, port=args.port)
    client._print_header()

    if not await client.connect():
        print(f"\n{client.BRIGHT_RED}Failed to connect to server.{client.RESET}")
        print(f"  {client.DIM}Expected: ws://{args.host}:{args.port}{client.RESET}")
        print(f"  {client.DIM}Make sure the Voice Agent server is running.{client.RESET}")
        return

    try:
        await asyncio.gather(
            client.handle_messages(),
            input_loop(client)
        )
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
