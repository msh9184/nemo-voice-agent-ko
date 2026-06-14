#!/usr/bin/env python3
"""
Audio File Client for Voice Agent - Professional STT Testing Tool

This client streams a WAV file to the Voice Agent server as if it were
real-time microphone input. Perfect for testing without a microphone.

Features:
- Streams WAV file at real-time rate (simulating live microphone)
- Supports adjustable playback speed for faster testing
- Compatible with STT-Only, Full Voice Agent modes
- Professional CLI output with clean transcription display
- Proper Protobuf frame serialization (same as browser client)
- Comprehensive session statistics

Usage:
    python audio_file_client.py --file audio.wav [options]

Examples:
    # Normal real-time streaming
    python audio_file_client.py --file test_audio.wav

    # 2x speed for faster testing
    python audio_file_client.py --file test_audio.wav --speed 2.0

    # Connect to specific server
    python audio_file_client.py --file test_audio.wav --host 10.0.0.1 --port 8765

Requirements:
    pip install websockets pipecat-ai
"""

import asyncio
import json
import uuid
import argparse
import sys
import wave
import struct
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

try:
    import websockets
except ImportError:
    print("Error: websockets package not installed.")
    print("Install with: pip install websockets")
    sys.exit(1)

try:
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.frames.frames import OutputAudioRawFrame
except ImportError:
    print("Error: pipecat-ai package not installed.")
    print("Install with: pip install pipecat-ai")
    sys.exit(1)


class ProfessionalDisplay:
    """Professional console display with clean formatting and real-time streaming."""

    # ANSI Escape Codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    # Colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Cursor control
    CLEAR_LINE = "\033[2K"
    CURSOR_UP = "\033[A"
    CURSOR_DOWN = "\033[B"
    SAVE_CURSOR = "\033[s"
    RESTORE_CURSOR = "\033[u"

    # Box drawing
    BOX_H = "═"
    BOX_V = "║"
    BOX_TL = "╔"
    BOX_TR = "╗"
    BOX_BL = "╚"
    BOX_BR = "╝"
    LINE = "─"
    THICK_LINE = "━"

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
        self._streaming_active = False
        self._last_partial_text = ""
        self._last_partial_len = 0
        self._progress_line = ""
        self._partial_line = ""

    def header(self, title: str, subtitle: str = ""):
        """Print application header."""
        width = 62
        print()
        print(f"{self.BRIGHT_CYAN}{self.BOX_TL}{self.BOX_H * width}{self.BOX_TR}{self.RESET}")
        print(f"{self.BRIGHT_CYAN}{self.BOX_V}{self.RESET}  {self.BOLD}{self.BRIGHT_WHITE}{title}{self.RESET}{' ' * (width - len(title) - 2)}{self.BRIGHT_CYAN}{self.BOX_V}{self.RESET}")
        if subtitle:
            print(f"{self.BRIGHT_CYAN}{self.BOX_V}{self.RESET}  {self.DIM}{subtitle}{self.RESET}{' ' * (width - len(subtitle) - 2)}{self.BRIGHT_CYAN}{self.BOX_V}{self.RESET}")
        print(f"{self.BRIGHT_CYAN}{self.BOX_BL}{self.BOX_H * width}{self.BOX_BR}{self.RESET}")
        print()

    def section(self, title: str, style: str = "normal"):
        """Print section header."""
        colors = {
            "normal": self.BRIGHT_CYAN,
            "success": self.BRIGHT_GREEN,
            "warning": self.YELLOW,
            "error": self.RED,
        }
        color = colors.get(style, self.BRIGHT_CYAN)
        print()
        print(f"  {color}{self.LINE * 58}{self.RESET}")
        print(f"  {self.BOLD}{self.BRIGHT_WHITE}{title}{self.RESET}")
        print(f"  {color}{self.LINE * 58}{self.RESET}")
        print()

    def log(self, level: str, message: str):
        """Print timestamped log message."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        icons = {
            "info": (self.CYAN, "ℹ"),
            "success": (self.BRIGHT_GREEN, "✓"),
            "warning": (self.YELLOW, "⚠"),
            "error": (self.RED, "✗"),
            "debug": (self.GRAY, "·"),
        }
        color, icon = icons.get(level, (self.WHITE, "•"))

        if self._streaming_active:
            # Move above streaming display area
            sys.stdout.write(f"{self.CURSOR_UP}{self.CURSOR_UP}")
            sys.stdout.write(f"\r{self.CLEAR_LINE}")
            print(f"{self.GRAY}{timestamp}{self.RESET} {color}[{icon}]{self.RESET} {message}")
            print()  # Progress line
            print()  # Partial line
            self._update_streaming_display()
        else:
            print(f"{self.GRAY}{timestamp}{self.RESET} {color}[{icon}]{self.RESET} {message}")

    def _clear_progress(self):
        """Clear streaming display if active (for backward compatibility)."""
        if self._streaming_active:
            self.end_streaming()

    def start_streaming(self):
        """Initialize streaming display mode with two-line area."""
        self._streaming_active = True
        self._last_partial_text = ""
        self._last_partial_len = 0
        # Reserve two lines for streaming display
        print()  # Progress line
        print()  # Partial transcription line
        sys.stdout.flush()

    def end_streaming(self):
        """End streaming display mode."""
        if self._streaming_active:
            # Clear both streaming lines
            sys.stdout.write(f"{self.CURSOR_UP}{self.CLEAR_LINE}")  # Clear partial line
            sys.stdout.write(f"{self.CURSOR_UP}{self.CLEAR_LINE}")  # Clear progress line
            sys.stdout.write("\r")
            sys.stdout.flush()
            self._streaming_active = False

    def transcription(self, text: str, is_final: bool, reason: str = "", index: int = 0):
        """Display transcription output with real-time streaming effect."""
        if is_final:
            # Format reason indicator
            reason_icons = {
                "boundary": ".",
                "VAD": "|",
                "EOU": "◆",
                "timeout": "⏱",
                "stream_end": "■",
            }
            icon = reason_icons.get(reason, "•")

            if self._streaming_active:
                # Move up 2 lines, print final, then restore streaming area
                sys.stdout.write(f"{self.CURSOR_UP}{self.CURSOR_UP}")
                sys.stdout.write(f"\r{self.CLEAR_LINE}")

            # Alternating colors for better readability
            num_color = self.BLUE if index % 2 == 0 else self.MAGENTA
            print(f"  {self.GREEN}✓{self.RESET} {num_color}[{index:02d}]{self.RESET} {text} {self.GRAY}{icon}{self.RESET}")

            if self._streaming_active:
                # Re-add streaming lines
                print()  # Progress line
                print()  # Partial line
                # Restore progress and partial displays
                self._update_streaming_display()

            self._last_partial_text = ""
            self._last_partial_len = 0
        else:
            # Partial (interim) - update the partial line dynamically
            self._last_partial_text = text
            if self._streaming_active:
                self._update_streaming_display()
            else:
                # Simple overwrite when not in streaming mode
                truncated = text[:70] + "..." if len(text) > 70 else text
                display = f"  {self.CYAN}▸{self.RESET} {self.DIM}{truncated}{self.RESET}"
                padding = " " * max(0, self._last_partial_len - len(truncated))
                sys.stdout.write(f"\r{display}{padding}")
                sys.stdout.flush()
                self._last_partial_len = len(truncated)

    def progress(self, current: int, total: int, elapsed: float):
        """Show progress indicator in streaming mode."""
        pct = current / total * 100
        bar_width = 20
        filled = int(bar_width * current / total)
        bar = "█" * filled + "░" * (bar_width - filled)

        self._progress_line = f"  {self.GRAY}[{bar}] {pct:5.1f}% │ {current:,}/{total:,} chunks │ {elapsed:.1f}s{self.RESET}"

        if self._streaming_active:
            self._update_streaming_display()
        else:
            sys.stdout.write(f"\r{self.CLEAR_LINE}{self._progress_line}")
            sys.stdout.flush()

    def _update_streaming_display(self):
        """Update both progress and partial lines atomically."""
        if not self._streaming_active:
            return

        # Move to progress line (2 lines up)
        sys.stdout.write(f"{self.CURSOR_UP}{self.CURSOR_UP}")

        # Draw progress line
        sys.stdout.write(f"\r{self.CLEAR_LINE}{self._progress_line}")
        sys.stdout.write("\n")

        # Draw partial transcription line
        if self._last_partial_text:
            truncated = self._last_partial_text[:75] + "..." if len(self._last_partial_text) > 75 else self._last_partial_text
            partial_display = f"  {self.CYAN}▸{self.RESET} {self.BRIGHT_WHITE}{truncated}{self.RESET}"
        else:
            partial_display = f"  {self.GRAY}(waiting for transcription...){self.RESET}"
        sys.stdout.write(f"\r{self.CLEAR_LINE}{partial_display}")
        sys.stdout.write("\n")

        sys.stdout.flush()

    def stats(self, data: Dict):
        """Display statistics."""
        print()
        print(f"  {self.GRAY}{self.LINE * 40}{self.RESET}")
        print(f"  {self.BOLD}Session Statistics{self.RESET}")
        for key, value in data.items():
            print(f"    {self.CYAN}{key}:{self.RESET} {value}")
        print(f"  {self.GRAY}{self.LINE * 40}{self.RESET}")

    def final_results(self, transcriptions: List[str], last_partial: str = ""):
        """Display final results summary."""
        self.section("Streaming Complete", "success")

        if transcriptions:
            print(f"  {self.BOLD}Final Transcriptions ({len(transcriptions)} utterances):{self.RESET}")
            print()
            for i, text in enumerate(transcriptions, 1):
                print(f"  {self.GRAY}{i:2d}.{self.RESET} {text}")
        else:
            print(f"  {self.YELLOW}No final transcriptions received.{self.RESET}")

        if last_partial:
            print()
            print(f"  {self.DIM}(Last partial - not finalized):{self.RESET}")
            print(f"  {self.DIM}    {last_partial}{self.RESET}")
        print()


class AudioFileClient:
    """Professional CLI client that streams audio files to Voice Agent server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        speed: float = 1.0,
        chunk_ms: int = 16,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.ws_url = f"ws://{host}:{port}"
        self.speed = speed
        self.chunk_ms = chunk_ms
        self.verbose = verbose

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.serializer = ProtobufFrameSerializer()
        self.connected = False
        self.running = True
        self.streaming = False

        # Display
        self.display = ProfessionalDisplay()

        # Server config
        self.service_mode = "unknown"
        self.stt_enabled = False
        self.config_received = False

        # Transcription state
        self.current_partial = ""
        self.final_transcriptions: List[str] = []
        self.transcription_count = 0

        # Stats
        self.bytes_sent = 0
        self.frames_sent = 0
        self.start_time = None

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize transcription text for display."""
        if not text:
            return text
        # Replace SentencePiece ▁ boundary with space
        text = text.replace('\u2581', ' ')
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        self.display.log("info", f"Connecting to {self.display.CYAN}{self.ws_url}{self.display.RESET}...")

        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
                max_size=10 * 1024 * 1024,
            )
            self.connected = True
            self.display.log("success", "WebSocket connection established")

            # Send client-ready with audio enabled
            await self._send_client_ready()
            return True

        except ConnectionRefusedError:
            self.display.log("error", "Connection refused - is the server running?")
            return False
        except Exception as e:
            self.display.log("error", f"Failed to connect: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the server."""
        self.running = False
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
        self.connected = False

    async def _send_client_ready(self):
        """Send RTVI client-ready message with audio enabled."""
        message = {
            "id": str(uuid.uuid4()),
            "type": "client-ready",
            "data": {
                "config": {
                    "audio_in_enabled": True,
                    "audio_out_enabled": True,
                }
            }
        }
        await self.websocket.send(json.dumps(message))

    async def _send_audio_frame(self, audio_data: bytes, sample_rate: int):
        """Send audio frame using Protobuf serialization."""
        frame = OutputAudioRawFrame(
            audio=audio_data,
            sample_rate=sample_rate,
            num_channels=1,
        )
        serialized = await self.serializer.serialize(frame)
        await self.websocket.send(serialized)
        self.bytes_sent += len(audio_data)
        self.frames_sent += 1

    def load_wav_file(self, file_path: str) -> tuple:
        """Load and validate WAV file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not path.suffix.lower() == ".wav":
            raise ValueError(f"Only WAV files are supported, got: {path.suffix}")

        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = wav.readframes(n_frames)

            duration = n_frames / sample_rate

            self.display.log("info", f"Loaded: {self.display.BOLD}{path.name}{self.display.RESET}")
            self.display.log("info", f"  Format: {sample_rate}Hz, {channels}ch, {sample_width*8}-bit")
            self.display.log("info", f"  Duration: {duration:.2f}s ({n_frames:,} samples)")

            # Validate format
            if sample_width != 2:
                raise ValueError(f"Expected 16-bit audio, got {sample_width*8}-bit")

            # Convert stereo to mono if needed
            if channels == 2:
                self.display.log("warning", "Converting stereo to mono...")
                samples = struct.unpack(f"<{n_frames * 2}h", audio_data)
                mono_samples = [
                    (samples[i] + samples[i + 1]) // 2
                    for i in range(0, len(samples), 2)
                ]
                audio_data = struct.pack(f"<{len(mono_samples)}h", *mono_samples)

            # Resample warning
            if sample_rate != 16000:
                self.display.log("warning", f"Sample rate is {sample_rate}Hz, server expects 16000Hz")
                self.display.log("warning", "Consider: ffmpeg -i input.wav -ar 16000 output.wav")

            return audio_data, sample_rate, duration

    async def stream_audio_file(self, file_path: str):
        """Stream audio file to server at real-time rate."""
        try:
            audio_data, sample_rate, duration = self.load_wav_file(file_path)
        except Exception as e:
            self.display.log("error", f"Failed to load audio: {e}")
            return

        # Calculate chunk size
        samples_per_chunk = int(sample_rate * self.chunk_ms / 1000)
        bytes_per_chunk = samples_per_chunk * 2  # 16-bit = 2 bytes per sample
        delay_per_chunk = (self.chunk_ms / 1000) / self.speed

        total_chunks = len(audio_data) // bytes_per_chunk
        effective_duration = duration / self.speed

        self.display.log("info", f"Streaming: {total_chunks:,} chunks at {self.speed}x speed")
        self.display.log("info", f"Estimated time: {effective_duration:.1f}s")

        self.display.section("Transcription Output")

        self.start_time = time.time()
        self.streaming = True
        self.display.start_streaming()  # Initialize real-time streaming display

        # Stream chunks
        progress_interval = max(1, total_chunks // 50)  # Update progress ~50 times

        for i in range(0, len(audio_data), bytes_per_chunk):
            if not self.running or not self.connected:
                break

            chunk = audio_data[i:i + bytes_per_chunk]
            if len(chunk) < bytes_per_chunk:
                chunk += b'\x00' * (bytes_per_chunk - len(chunk))

            await self._send_audio_frame(chunk, sample_rate)

            chunks_sent = (i // bytes_per_chunk) + 1
            if chunks_sent % progress_interval == 0 or chunks_sent == total_chunks:
                elapsed = time.time() - self.start_time
                self.display.progress(chunks_sent, total_chunks, elapsed)

            await asyncio.sleep(delay_per_chunk)

        self.display.end_streaming()  # Clean up streaming display
        self.streaming = False

        # Wait for final transcriptions with proper handling
        self.display.log("info", "Waiting for final transcriptions...")

        # Wait in increments, checking for new transcriptions
        wait_time = 3.0
        check_interval = 0.5
        last_count = len(self.final_transcriptions)

        for _ in range(int(wait_time / check_interval)):
            await asyncio.sleep(check_interval)
            current_count = len(self.final_transcriptions)
            if current_count > last_count:
                last_count = current_count
                # Reset wait timer if we received new transcription
                continue

        # Include last partial if not finalized
        if self.current_partial and self.current_partial not in self.final_transcriptions:
            # Check if the partial is substantially different from last final
            if not self.final_transcriptions or not self._is_duplicate(self.current_partial, self.final_transcriptions[-1]):
                self.final_transcriptions.append(self.current_partial)
                self.transcription_count += 1
                self.display.transcription(self.current_partial, True, "stream_end", self.transcription_count)

        # Show final results
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.display.final_results(self.final_transcriptions)

        # Show stats
        total_chars = sum(len(t) for t in self.final_transcriptions)
        self.display.stats({
            "Utterances": len(self.final_transcriptions),
            "Characters": f"{total_chars:,}",
            "Audio Duration": f"{duration:.1f}s",
            "Processing Time": f"{elapsed:.1f}s",
            "Data Sent": f"{self.bytes_sent/1024:.1f} KB ({self.frames_sent:,} frames)",
            "Speed": f"{self.speed}x (effective: {duration/elapsed:.2f}x)" if elapsed > 0 else f"{self.speed}x",
        })

    def _is_duplicate(self, text1: str, text2: str) -> bool:
        """Check if two texts are substantially similar."""
        if not text1 or not text2:
            return False
        # Normalize and compare
        t1 = self.normalize_text(text1).lower()
        t2 = self.normalize_text(text2).lower()
        # Check for significant overlap
        if t1 == t2:
            return True
        if t1.startswith(t2) or t2.startswith(t1):
            diff = abs(len(t1) - len(t2))
            return diff < 20  # Small difference = duplicate
        return False

    async def handle_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    if isinstance(message, str):
                        data = json.loads(message)
                        await self._process_message(data)
                    elif isinstance(message, bytes):
                        json_objects = self._extract_json_from_binary(message)
                        for data in json_objects:
                            await self._process_message(data)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    if self.verbose:
                        self.display.log("debug", f"Message error: {e}")

        except websockets.exceptions.ConnectionClosed:
            self.connected = False
        except Exception as e:
            if self.verbose:
                self.display.log("error", f"Message handler error: {e}")
            self.connected = False

    def _extract_json_from_binary(self, data: bytes) -> list:
        """Extract JSON from binary data."""
        results = []
        try:
            text = data.decode('utf-8')
            results.append(json.loads(text))
            return results
        except:
            pass

        try:
            data_str = data.decode('latin-1')
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
                            obj = json.loads(data_str[i:j])
                            results.append(obj)
                            i = j
                            continue
                        except:
                            pass
                i += 1
        except:
            pass

        return results

    async def _process_message(self, data: dict):
        """Process RTVI message."""
        msg_type = data.get("type", "")
        msg_data = data.get("data", {})

        if msg_type == "bot-ready":
            self.display.log("success", "Bot is ready")

        elif msg_type == "server-config":
            self.service_mode = msg_data.get("service_mode", "unknown")
            stt_config = msg_data.get("stt", {})
            self.stt_enabled = stt_config.get("enabled", False)
            self.config_received = True
            mode_str = f"{self.display.CYAN}{self.service_mode}{self.display.RESET}"
            stt_str = f"{self.display.GREEN}enabled{self.display.RESET}" if self.stt_enabled else f"{self.display.RED}disabled{self.display.RESET}"
            self.display.log("info", f"Server mode: {mode_str}, STT: {stt_str}")

        elif msg_type == "user-transcription":
            raw_text = msg_data.get("text", "")
            final = msg_data.get("final", False)
            result = msg_data.get("result", {})
            reason = result.get("reason", "") if isinstance(result, dict) else ""

            if raw_text:
                text = self.normalize_text(raw_text)

                if final:
                    # Check for duplicate
                    if text not in self.final_transcriptions:
                        self.final_transcriptions.append(text)
                        self.transcription_count += 1
                        self.display.transcription(text, True, reason, self.transcription_count)
                    self.current_partial = ""
                else:
                    self.current_partial = text
                    # Always show partial for dynamic real-time streaming effect
                    self.display.transcription(text, False)

        elif msg_type == "user-started-speaking":
            if self.verbose and self.streaming:
                self.display.log("debug", "VAD: Speech started")

        elif msg_type == "user-stopped-speaking":
            if self.verbose and self.streaming:
                self.display.log("debug", "VAD: Speech ended")


async def main():
    parser = argparse.ArgumentParser(
        description="Stream audio files to Voice Agent server for STT testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Normal real-time streaming
    python audio_file_client.py --file test_audio.wav

    # 2x speed for faster testing
    python audio_file_client.py --file test_audio.wav --speed 2.0

    # Connect to remote server
    python audio_file_client.py --file audio.wav --host 10.0.0.1

WAV File Requirements:
    - Format: PCM (uncompressed)
    - Sample Rate: 16000Hz recommended
    - Bit Depth: 16-bit required
    - Channels: Mono or Stereo (stereo will be converted)

Convert audio with ffmpeg:
    ffmpeg -i input.mp3 -ar 16000 -ac 1 -acodec pcm_s16le output.wav
"""
    )
    parser.add_argument("--file", "-f", required=True, help="Path to WAV audio file")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (default: 1.0)")
    parser.add_argument("--chunk-ms", type=int, default=16, help="Chunk size in ms (default: 16)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")

    args = parser.parse_args()

    client = AudioFileClient(
        host=args.host,
        port=args.port,
        speed=args.speed,
        chunk_ms=args.chunk_ms,
        verbose=args.verbose,
    )

    client.display.header(
        "Audio File Client - STT Testing Tool",
        "Stream WAV files to test speech recognition"
    )

    if not await client.connect():
        print(f"\n{client.display.RED}Failed to connect to server.{client.display.RESET}")
        print(f"  {client.display.DIM}Ensure Voice Agent server is running at {args.host}:{args.port}{client.display.RESET}\n")
        return

    try:
        # Start message handler in background
        message_task = asyncio.create_task(client.handle_messages())

        # Wait for config
        await asyncio.sleep(0.5)

        # Stream audio file
        await client.stream_audio_file(args.file)

        # Cancel message handler
        message_task.cancel()
        try:
            await message_task
        except asyncio.CancelledError:
            pass

    except KeyboardInterrupt:
        print(f"\n{client.display.YELLOW}Interrupted{client.display.RESET}")
    except Exception as e:
        print(f"\n{client.display.RED}Error: {e}{client.display.RESET}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
