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
Real-time console display utility for STT output.

Provides clean, readable console output for streaming speech-to-text:
- Overwrites current line for interim (partial) transcriptions
- Adds newline for final transcriptions
- Supports colored output for better readability
"""

import sys
import time
from typing import Optional


class RealtimeDisplay:
    """
    Real-time console display for STT transcription.

    Features:
    - Overwrite mode: Updates current line without creating new lines
    - Final mode: Commits the line and moves to next line
    - Color support: Different colors for interim vs final text
    - Timestamp support: Optional timestamps for each utterance
    """

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"

    def __init__(
        self,
        show_timestamps: bool = True,
        show_status: bool = True,
        use_colors: bool = True,
        min_display_length: int = 80,
        utterance_prefix: str = ">",
        final_prefix: str = "+",
    ):
        """
        Initialize the real-time display.

        Args:
            show_timestamps: Show timestamps for each utterance
            show_status: Show interim/final status indicator
            use_colors: Use ANSI colors for output
            min_display_length: Minimum line length (for clean overwriting)
            utterance_prefix: Prefix for interim utterances
            final_prefix: Prefix for final utterances
        """
        self.show_timestamps = show_timestamps
        self.show_status = show_status
        self.use_colors = use_colors
        self.min_display_length = min_display_length
        self.utterance_prefix = utterance_prefix
        self.final_prefix = final_prefix

        self._current_text = ""
        self._last_display_length = 0
        self._utterance_count = 0
        self._start_time: Optional[float] = None

    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        if not self.show_timestamps:
            return ""
        return time.strftime("%H:%M:%S")

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{color}{text}{self.RESET}"

    def _clear_line(self):
        """Clear the current line."""
        # Move to start of line and clear
        sys.stdout.write("\r" + " " * self._last_display_length + "\r")
        sys.stdout.flush()

    def update_interim(self, text: str):
        """
        Update the display with interim (partial) transcription.
        Overwrites the current line without creating a new line.

        Args:
            text: The interim transcription text
        """
        if not text or not text.strip():
            return

        text = text.strip()

        # Don't update if text hasn't changed
        if text == self._current_text:
            return

        self._current_text = text

        # Build the display line
        timestamp = self._get_timestamp()
        prefix = self.utterance_prefix

        if self.use_colors:
            # Interim: cyan text, dim
            display = f"\r{self.GRAY}{timestamp}{self.RESET} {self.CYAN}{prefix}{self.RESET} {self.DIM}{text}{self.RESET}"
        else:
            display = f"\r{timestamp} {prefix} {text}"

        # Pad to ensure clean overwrite
        display_length = len(timestamp) + len(prefix) + len(text) + 4
        if display_length < self._last_display_length:
            display += " " * (self._last_display_length - display_length)

        self._last_display_length = max(display_length, self._last_display_length)

        # Write without newline (overwrite mode)
        sys.stdout.write(display)
        sys.stdout.flush()

    def commit_final(self, text: str):
        """
        Commit a final transcription and move to a new line.

        Args:
            text: The final transcription text
        """
        if not text or not text.strip():
            # If no text, just clear the line
            self._clear_line()
            self._current_text = ""
            return

        text = text.strip()
        self._utterance_count += 1

        # Clear the interim line first
        self._clear_line()

        # Build the final display line
        timestamp = self._get_timestamp()
        prefix = self.final_prefix
        count = f"[{self._utterance_count}]"

        if self.use_colors:
            # Final: green text, bold
            line = f"{self.GRAY}{timestamp}{self.RESET} {self.GREEN}{prefix}{self.RESET} {count} {self.BOLD}{self.WHITE}{text}{self.RESET}"
        else:
            line = f"{timestamp} {prefix} {count} {text}"

        # Print with newline
        print(line)
        sys.stdout.flush()

        # Reset state for next utterance
        self._current_text = ""
        self._last_display_length = 0

    def reset(self):
        """Reset the display state."""
        self._clear_line()
        self._current_text = ""
        self._last_display_length = 0

    def print_separator(self, char: str = "─", length: int = 50):
        """Print a separator line."""
        if self.use_colors:
            print(f"{self.GRAY}{char * length}{self.RESET}")
        else:
            print(char * length)

    def print_header(self, text: str):
        """Print a header message."""
        self.print_separator()
        if self.use_colors:
            print(f"{self.BOLD}{self.CYAN}{text}{self.RESET}")
        else:
            print(text)
        self.print_separator()


class VoiceAgentDisplay:
    """
    Enhanced real-time console display for Voice Agent pipeline.

    Shows clean, readable output for:
    - STT: Partial (dim) and Final (bold) transcriptions
    - LLM: Streaming responses
    - TTS: Text being synthesized

    Example output:
    ──────────────────────────────────────────────────
    [*] Voice Agent Active
    ──────────────────────────────────────────────────
    14:25:01 > 안녕하세요                              (partial, overwrites)
    14:25:02 + [1] 안녕하세요                          (final, newline)
    14:25:03 [AI] 네, 안녕하세요! 무엇을 도와드릴까요? (LLM response)
    14:25:04 [TTS] 네, 안녕하세요! 무엇을...           (TTS chunk)
    """

    # ANSI codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"

    def __init__(
        self,
        use_colors: bool = True,
        show_timestamps: bool = True,
        max_tts_display_len: int = 50,
    ):
        self.use_colors = use_colors
        self.show_timestamps = show_timestamps
        self.max_tts_display_len = max_tts_display_len

        # STT state
        self._stt_current_text = ""
        self._stt_last_display_len = 0
        self._stt_utterance_count = 0

        # LLM state
        self._llm_accumulated_text = ""
        self._llm_response_count = 0

        # TTS state
        self._tts_current_chunk = ""

    def _get_timestamp(self) -> str:
        if not self.show_timestamps:
            return ""
        return time.strftime("%H:%M:%S")

    def _colorize(self, text: str, color: str) -> str:
        if not self.use_colors:
            return text
        return f"{color}{text}{self.RESET}"

    def _clear_line(self):
        sys.stdout.write("\r" + " " * self._stt_last_display_len + "\r")
        sys.stdout.flush()

    # ─────────────────────────────────────────────────
    # STT Display
    # ─────────────────────────────────────────────────

    def stt_interim(self, text: str):
        """Display STT partial result (overwrites current line)."""
        if not text or not text.strip():
            return

        text = text.strip()
        if text == self._stt_current_text:
            return

        self._stt_current_text = text
        ts = self._get_timestamp()

        if self.use_colors:
            display = f"\r{self.GRAY}{ts}{self.RESET} {self.CYAN}>{self.RESET} {self.DIM}{text}{self.RESET}"
        else:
            display = f"\r{ts} > {text}"

        display_len = len(ts) + len(text) + 5
        if display_len < self._stt_last_display_len:
            display += " " * (self._stt_last_display_len - display_len)
        self._stt_last_display_len = max(display_len, self._stt_last_display_len)

        sys.stdout.write(display)
        sys.stdout.flush()

    def stt_final(self, text: str, reason: str = ""):
        """Display STT final result (commits with newline)."""
        if not text or not text.strip():
            self._clear_line()
            self._stt_current_text = ""
            return

        text = text.strip()
        self._stt_utterance_count += 1
        self._clear_line()

        ts = self._get_timestamp()
        count = f"[{self._stt_utterance_count}]"
        reason_str = f" ({reason})" if reason else ""

        if self.use_colors:
            line = f"{self.GRAY}{ts}{self.RESET} {self.GREEN}+{self.RESET} {count} {self.BOLD}{self.WHITE}{text}{self.RESET}{self.GRAY}{reason_str}{self.RESET}"
        else:
            line = f"{ts} + {count} {text}{reason_str}"

        print(line)
        sys.stdout.flush()

        self._stt_current_text = ""
        self._stt_last_display_len = 0

    # ─────────────────────────────────────────────────
    # LLM Display
    # ─────────────────────────────────────────────────

    def llm_start(self):
        """Signal LLM response starting."""
        self._llm_accumulated_text = ""
        self._llm_response_count += 1

    def llm_chunk(self, text: str):
        """Display LLM streaming chunk (accumulates)."""
        self._llm_accumulated_text += text

    def llm_end(self):
        """Display complete LLM response."""
        if not self._llm_accumulated_text.strip():
            return

        ts = self._get_timestamp()
        text = self._llm_accumulated_text.strip()

        if self.use_colors:
            line = f"{self.GRAY}{ts}{self.RESET} {self.MAGENTA}[AI]{self.RESET} {self.WHITE}{text}{self.RESET}"
        else:
            line = f"{ts} [AI] {text}"

        print(line)
        sys.stdout.flush()
        self._llm_accumulated_text = ""

    # ─────────────────────────────────────────────────
    # TTS Display
    # ─────────────────────────────────────────────────

    def tts_chunk(self, text: str):
        """Display TTS text chunk being synthesized."""
        if not text or not text.strip():
            return

        text = text.strip()
        ts = self._get_timestamp()

        # Truncate long text for display
        display_text = text[:self.max_tts_display_len] + "..." if len(text) > self.max_tts_display_len else text

        if self.use_colors:
            line = f"{self.GRAY}{ts}{self.RESET} {self.BLUE}[TTS]{self.RESET} {self.DIM}{display_text}{self.RESET}"
        else:
            line = f"{ts} [TTS] {display_text}"

        print(line)
        sys.stdout.flush()

    # ─────────────────────────────────────────────────
    # Text Input Mode Display (for LLM-Only, TTS-Only, LLM+TTS)
    # ─────────────────────────────────────────────────

    def user_text(self, text: str):
        """Display user text input (for text input modes)."""
        if not text or not text.strip():
            return

        text = text.strip()
        ts = self._get_timestamp()

        if self.use_colors:
            line = f"{self.GRAY}{ts}{self.RESET} {self.CYAN}[USER]{self.RESET} {self.WHITE}{text}{self.RESET}"
        else:
            line = f"{ts} [USER] {text}"

        print(line)
        sys.stdout.flush()

    def bot_text(self, text: str):
        """Display bot text response (for text input modes)."""
        if not text or not text.strip():
            return

        text = text.strip()
        ts = self._get_timestamp()

        if self.use_colors:
            line = f"{self.GRAY}{ts}{self.RESET} {self.MAGENTA}[BOT]{self.RESET} {self.WHITE}{text}{self.RESET}"
        else:
            line = f"{ts} [BOT] {text}"

        print(line)
        sys.stdout.flush()

    # ─────────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────────

    def separator(self, char: str = "─", length: int = 50):
        """Print a separator line."""
        if self.use_colors:
            print(f"{self.GRAY}{char * length}{self.RESET}")
        else:
            print(char * length)

    def header(self, text: str):
        """Print a header."""
        self.separator()
        if self.use_colors:
            print(f"{self.BOLD}{self.CYAN}{text}{self.RESET}")
        else:
            print(text)
        self.separator()

    def reset(self):
        """Reset all state."""
        self._clear_line()
        self._stt_current_text = ""
        self._stt_last_display_len = 0
        self._llm_accumulated_text = ""


# Global instance for easy access
_display: Optional[RealtimeDisplay] = None
_voice_agent_display: Optional[VoiceAgentDisplay] = None


def get_display() -> RealtimeDisplay:
    """Get the global display instance."""
    global _display
    if _display is None:
        _display = RealtimeDisplay()
    return _display


def set_display(display: RealtimeDisplay):
    """Set the global display instance."""
    global _display
    _display = display


def get_voice_agent_display() -> VoiceAgentDisplay:
    """Get the global Voice Agent display instance."""
    global _voice_agent_display
    if _voice_agent_display is None:
        _voice_agent_display = VoiceAgentDisplay()
    return _voice_agent_display


def set_voice_agent_display(display: VoiceAgentDisplay):
    """Set the global Voice Agent display instance."""
    global _voice_agent_display
    _voice_agent_display = display


def display_interim(text: str):
    """Convenience function for interim display."""
    get_display().update_interim(text)


def display_final(text: str):
    """Convenience function for final display."""
    get_display().commit_final(text)


def display_reset():
    """Convenience function to reset display."""
    get_display().reset()
