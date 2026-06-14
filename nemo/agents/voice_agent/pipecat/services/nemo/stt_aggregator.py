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
STT Text Aggregator for STT-only mode.

Manages text buffer accumulation and sentence boundary detection for clean
transcription output. Handles prefix stripping to prevent overlap between
consecutive sentences.
"""

import asyncio
import re
import sys
import time
from typing import Optional, Set

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    EndTaskFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


class RealtimeConsoleDisplay:
    """Professional real-time console display with clean output management.

    Provides real-time streaming display for STT transcription:
    - Interim (partial) results: Overwrites current line for dynamic streaming effect
    - Final results: Commits to a new line with utterance numbering
    """

    # ANSI escape codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"

    # Cursor control
    CURSOR_UP = "\033[A"
    CLEAR_LINE = "\033[2K"
    CURSOR_START = "\r"

    def __init__(self, use_colors: bool = True, show_interim: bool = True):
        """Initialize display.

        Args:
            use_colors: Enable ANSI color codes
            show_interim: Show interim (partial) transcriptions for real-time streaming effect
        """
        self.use_colors = use_colors
        self._show_interim = show_interim
        self._last_line_length = 0
        self._utterance_count = 0
        self._has_pending_interim = False
        self._session_start_time = time.time()
        self._total_chars = 0
        self._reason_counts = {}

    def _clear_current_line(self):
        """Clear current line using ANSI escape."""
        if self.use_colors:
            sys.stdout.write(f"{self.CURSOR_START}{self.CLEAR_LINE}")
        else:
            sys.stdout.write("\r" + " " * max(self._last_line_length, 120) + "\r")
        sys.stdout.flush()
        self._last_line_length = 0

    def _format_reason(self, reason: str) -> str:
        """Format reason indicator with icons."""
        reason_icons = {
            "boundary": ".",     # Punctuation boundary
            "VAD": "|",          # Voice activity detection
            "EOU": "◆",          # End of utterance token
            "timeout": "⏱",      # Silence timeout
            "stream_end": "■",   # Stream ended
        }
        icon = reason_icons.get(reason, "•")

        if self.use_colors:
            return f"{self.GRAY}{icon}{self.RESET}"
        return icon

    def show_interim(self, text: str):
        """Display interim text (disabled by default for clean output)."""
        if not self._show_interim or not text:
            return

        self._clear_current_line()

        # Truncate long text for display
        display_text = text[:80] + "..." if len(text) > 80 else text

        if self.use_colors:
            display = f"{self.CYAN}▸{self.RESET} {self.DIM}{display_text}{self.RESET}"
        else:
            display = f"> {display_text}"

        sys.stdout.write(display)
        sys.stdout.flush()
        self._last_line_length = len(display_text) + 3
        self._has_pending_interim = True

    def show_final(self, text: str, reason: str = ""):
        """Display final transcription with formatting."""
        if not text:
            self._clear_current_line()
            self._has_pending_interim = False
            return

        # Clear any pending interim display
        if self._has_pending_interim:
            self._clear_current_line()
            self._has_pending_interim = False

        self._utterance_count += 1
        self._total_chars += len(text)

        # Track reason statistics
        if reason:
            self._reason_counts[reason] = self._reason_counts.get(reason, 0) + 1

        # Format the output line
        reason_icon = self._format_reason(reason) if reason else ""

        if self.use_colors:
            # Professional format: icon [number] text (reason)
            num_color = self.BLUE if self._utterance_count % 2 == 0 else self.MAGENTA
            line = f"{self.GREEN}✓{self.RESET} {num_color}[{self._utterance_count:02d}]{self.RESET} {text} {reason_icon}"
        else:
            line = f"+ [{self._utterance_count:02d}] {text} ({reason})" if reason else f"+ [{self._utterance_count:02d}] {text}"

        print(line)
        sys.stdout.flush()
        self._last_line_length = 0

    def show_timeout_flush(self, text: str):
        """Display timeout-triggered flush."""
        if not text:
            self._clear_current_line()
            self._has_pending_interim = False
            return

        if self._has_pending_interim:
            self._clear_current_line()
            self._has_pending_interim = False

        self._utterance_count += 1
        self._total_chars += len(text)
        self._reason_counts["timeout"] = self._reason_counts.get("timeout", 0) + 1

        if self.use_colors:
            num_color = self.BLUE if self._utterance_count % 2 == 0 else self.MAGENTA
            line = f"{self.YELLOW}⏱{self.RESET} {num_color}[{self._utterance_count:02d}]{self.RESET} {text}"
        else:
            line = f"⏱ [{self._utterance_count:02d}] {text} (timeout)"

        print(line)
        sys.stdout.flush()
        self._last_line_length = 0

    def reset(self):
        """Reset display state (clears any pending partial)."""
        if self._has_pending_interim:
            self._clear_current_line()
        self._has_pending_interim = False
        self._last_line_length = 0

    def get_stats(self) -> dict:
        """Get session statistics."""
        elapsed = time.time() - self._session_start_time
        return {
            "utterances": self._utterance_count,
            "total_chars": self._total_chars,
            "elapsed_secs": elapsed,
            "chars_per_sec": self._total_chars / elapsed if elapsed > 0 else 0,
            "reason_counts": self._reason_counts.copy(),
        }

    def show_summary(self):
        """Display session summary."""
        stats = self.get_stats()

        if self.use_colors:
            print(f"\n{self.GRAY}{'─' * 50}{self.RESET}")
            print(f"{self.BOLD}Session Summary{self.RESET}")
            print(f"  {self.CYAN}Utterances:{self.RESET} {stats['utterances']}")
            print(f"  {self.CYAN}Characters:{self.RESET} {stats['total_chars']:,}")
            print(f"  {self.CYAN}Duration:{self.RESET} {stats['elapsed_secs']:.1f}s")
            if stats['reason_counts']:
                reasons = ", ".join(f"{k}:{v}" for k, v in stats['reason_counts'].items())
                print(f"  {self.CYAN}Boundaries:{self.RESET} {reasons}")
            print(f"{self.GRAY}{'─' * 50}{self.RESET}")
        else:
            print(f"\n{'─' * 50}")
            print(f"Summary: {stats['utterances']} utterances, {stats['total_chars']} chars, {stats['elapsed_secs']:.1f}s")
            print(f"{'─' * 50}")


class STTTextAggregator(FrameProcessor):
    """
    Text aggregator for STT-only mode with prefix stripping.

    Key features:
    - Sentence boundary detection (punctuation, VAD, EOU tokens)
    - Prefix stripping to prevent overlap between consecutive sentences
    - Real-time console display with line overwriting
    - Silence timeout for long pauses
    """

    def __init__(
        self,
        eou_string: str = "<EOU>",
        eob_string: str = "<EOB>",
        sentence_end_punctuation: str = ".?!。？！",
        use_vad_boundary: bool = True,
        use_punctuation_boundary: bool = True,
        min_words_for_punctuation_boundary: int = 3,
        max_words_before_commit: int = 20,
        silence_timeout_secs: float = 3.0,
        silence_check_interval_secs: float = 0.5,
        language: Language = Language.EN_US,
        use_realtime_display: bool = True,
        use_colors: bool = True,
        show_interim: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.eou_string = eou_string
        self.eob_string = eob_string
        self.sentence_end_punctuation: Set[str] = set(sentence_end_punctuation)
        self.use_vad_boundary = use_vad_boundary
        self.use_punctuation_boundary = use_punctuation_boundary
        self.min_words_for_punctuation_boundary = min_words_for_punctuation_boundary
        self.max_words_before_commit = max_words_before_commit
        self.silence_timeout_secs = silence_timeout_secs
        self.silence_check_interval_secs = silence_check_interval_secs
        self.language = language

        # State management
        self._text_buffer: str = ""
        self._last_displayed_text: str = ""
        self._vad_user_speaking: bool = False
        self._utterance_count: int = 0

        # Prefix stripping for overlap prevention
        self._committed_prefix: str = ""
        self._committed_prefix_timestamp: float = 0.0
        self._prefix_validity_secs: float = 60.0

        # Duplicate prevention
        self._last_committed_text: str = ""
        self._last_commit_time: float = 0.0
        self._duplicate_window_secs: float = 2.0

        # Silence monitoring
        self._last_activity_time: float = time.time()
        self._timeout_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Display with real-time streaming effect
        self._display = RealtimeConsoleDisplay(use_colors=use_colors, show_interim=show_interim) if use_realtime_display else None

        logger.info(f"[Aggregator] Initialized: VAD={use_vad_boundary}, Punct={use_punctuation_boundary}, Timeout={silence_timeout_secs}s, ShowInterim={show_interim}")

    # =========================================================================
    # Text Normalization
    # =========================================================================

    def _strip_special_tokens(self, text: str) -> str:
        """Remove EOU/EOB tokens and normalize text."""
        text = text.replace(self.eou_string, "").replace(self.eob_string, "")
        text = text.replace("\u2581", " ")  # SentencePiece marker
        text = text.replace("\uFFFD", "")   # Unicode replacement
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for prefix comparison."""
        text = text.replace("\u2581", " ")
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def _normalize_word(self, word: str) -> str:
        """Strip punctuation from word for comparison."""
        return re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word, flags=re.UNICODE)

    # =========================================================================
    # Prefix Stripping
    # =========================================================================

    def _strip_committed_prefix(self, text: str) -> str:
        """Strip committed prefix from incoming text to prevent overlap."""
        if not self._committed_prefix:
            return text

        elapsed = time.time() - self._committed_prefix_timestamp
        if elapsed > self._prefix_validity_secs:
            logger.debug(f"[Prefix] Expired after {elapsed:.1f}s")
            self._committed_prefix = ""
            return text

        normalized_text = self._normalize_for_comparison(text)
        text_words = normalized_text.split()
        prefix_words = self._committed_prefix.split()

        if not prefix_words:
            return text

        # Word-level matching with punctuation normalization
        match_count = 0
        for i, prefix_word in enumerate(prefix_words):
            if i >= len(text_words):
                break
            if text_words[i] == prefix_word:
                match_count += 1
            elif self._normalize_word(text_words[i]) == self._normalize_word(prefix_word):
                match_count += 1
            else:
                break

        if len(text_words) < len(prefix_words):
            return text

        match_ratio = match_count / len(prefix_words) if prefix_words else 0

        if match_ratio >= 0.7:
            original_words = text.split()
            if match_count < len(original_words):
                stripped = " ".join(original_words[match_count:]).strip()
                if stripped:
                    logger.debug(f"[Prefix] Stripped {match_count} words → '{stripped[:40]}...'")
                    return stripped
                return ""
            return ""

        return text

    def _set_committed_prefix(self, text: str, reason: str):
        """Set committed text as prefix for subsequent stripping."""
        if not text:
            return
        self._committed_prefix = self._normalize_for_comparison(text)
        self._committed_prefix_timestamp = time.time()
        logger.debug(f"[Prefix] Set ({reason}): '{text[:40]}...'")

    def _clear_committed_prefix(self, reason: str):
        """Clear committed prefix."""
        if self._committed_prefix:
            logger.debug(f"[Prefix] Cleared ({reason})")
        self._committed_prefix = ""
        self._committed_prefix_timestamp = 0.0

    # =========================================================================
    # Boundary Detection
    # =========================================================================

    def _detect_eou(self, text: str) -> bool:
        """Check for end of utterance tokens."""
        return self.eou_string in text or self.eob_string in text

    def _detect_sentence_boundary(self, text: str) -> bool:
        """Check if text ends with sentence boundary punctuation."""
        clean_text = self._strip_special_tokens(text).strip()
        if not clean_text:
            return False

        word_count = len(clean_text.split())

        if self.use_punctuation_boundary:
            if clean_text[-1] in self.sentence_end_punctuation:
                if word_count >= self.min_words_for_punctuation_boundary:
                    return True

        return False

    # =========================================================================
    # Duplicate Prevention
    # =========================================================================

    def _is_duplicate_commit(self, text: str) -> bool:
        """Check if text was recently committed."""
        if not self._last_committed_text:
            return False

        time_since_last = time.time() - self._last_commit_time
        if time_since_last >= self._duplicate_window_secs:
            return False

        if text == self._last_committed_text:
            return True

        if len(text) > len(self._last_committed_text):
            if text.startswith(self._last_committed_text):
                new_part = text[len(self._last_committed_text):].strip()
                if len(new_part.split()) < 5:
                    return True

        if len(self._last_committed_text) >= len(text):
            if self._last_committed_text.startswith(text):
                return True

        return False

    # =========================================================================
    # Commit Handlers
    # =========================================================================

    async def _commit_final(self, text: str, reason: str):
        """Commit final transcription and reset buffer."""
        clean_text = self._strip_special_tokens(text).strip()

        if not clean_text:
            self._reset_buffer()
            return

        if self._is_duplicate_commit(clean_text):
            self._reset_buffer()
            return

        self._last_committed_text = clean_text
        self._last_commit_time = time.time()
        self._utterance_count += 1

        if self._display:
            self._display.show_final(clean_text, reason)
        else:
            logger.info(f"[FINAL #{self._utterance_count}] ({reason}) {clean_text}")

        frame = TranscriptionFrame(
            text=clean_text,
            user_id="",
            timestamp=time_now_iso8601(),
            language=self.language,
            result={"text": clean_text, "is_final": True, "reason": reason},
        )
        await self.push_frame(frame)

        if reason != "VAD":
            self._set_committed_prefix(clean_text, reason)
        else:
            self._clear_committed_prefix("VAD_commit")

        self._reset_buffer()

    async def _commit_timeout(self, text: str):
        """Commit transcription due to silence timeout."""
        clean_text = self._strip_special_tokens(text).strip()

        if not clean_text or self._is_duplicate_commit(clean_text):
            self._reset_buffer()
            return

        self._last_committed_text = clean_text
        self._last_commit_time = time.time()
        self._utterance_count += 1

        if self._display:
            self._display.show_timeout_flush(clean_text)
        else:
            logger.info(f"[TIMEOUT #{self._utterance_count}] {clean_text}")

        frame = TranscriptionFrame(
            text=clean_text,
            user_id="",
            timestamp=time_now_iso8601(),
            language=self.language,
            result={"text": clean_text, "is_final": True, "reason": "timeout"},
        )
        await self.push_frame(frame)

        self._set_committed_prefix(clean_text, "timeout")
        self._reset_buffer()

    async def _update_interim(self, text: str):
        """Update interim display."""
        clean_text = self._strip_special_tokens(text).strip()

        if not clean_text or clean_text == self._last_displayed_text:
            return

        self._last_displayed_text = clean_text

        if self._display:
            self._display.show_interim(clean_text)

        frame = InterimTranscriptionFrame(
            text=clean_text,
            user_id="",
            timestamp=time_now_iso8601(),
            language=self.language,
            result={"text": clean_text, "is_final": False},
        )
        await self.push_frame(frame)

    def _reset_buffer(self):
        """Reset text buffer state."""
        self._text_buffer = ""
        self._last_displayed_text = ""
        if self._display:
            self._display.reset()

    # =========================================================================
    # Silence Monitoring
    # =========================================================================

    async def _silence_timeout_monitor(self):
        """Monitor silence and flush buffer on timeout."""
        while self._running:
            try:
                await asyncio.sleep(self.silence_check_interval_secs)

                if self.silence_timeout_secs <= 0 or self._vad_user_speaking:
                    continue

                if self._text_buffer.strip():
                    elapsed = time.time() - self._last_activity_time
                    if elapsed >= self.silence_timeout_secs:
                        await self._commit_timeout(self._text_buffer)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Aggregator] Silence monitor error: {e}")

    # =========================================================================
    # Frame Handlers
    # =========================================================================

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame, direction: FrameDirection
    ):
        """Handle incoming transcription frames."""
        self._last_activity_time = time.time()

        frame_reason = ""
        if hasattr(frame, 'result') and isinstance(frame.result, dict):
            frame_reason = frame.result.get('reason', '')

        raw_text = frame.text
        clean_text = self._strip_special_tokens(raw_text).strip()

        if not clean_text:
            return

        # Check EOU tokens first
        if self._detect_eou(raw_text):
            await self._commit_final(clean_text, "EOU")
            return

        # Strip committed prefix
        processed_text = clean_text
        if self._committed_prefix:
            stripped = self._strip_committed_prefix(clean_text)
            if not stripped:
                return  # Duplicate of committed text
            if stripped != clean_text:
                processed_text = stripped

        # Handle upstream final decisions
        if isinstance(frame, TranscriptionFrame):
            await self._commit_final(processed_text, frame_reason or "STT")
            return

        if frame_reason == "boundary":
            await self._commit_final(processed_text, "boundary")
            return

        # Check our own boundary detection
        if self._detect_sentence_boundary(processed_text):
            await self._commit_final(processed_text, "boundary")
            return

        # Buffer and display interim
        self._text_buffer = processed_text
        await self._update_interim(processed_text)

    async def _handle_vad_started(self, frame: VADUserStartedSpeakingFrame, direction: FrameDirection):
        """Handle VAD user started speaking."""
        self._vad_user_speaking = True
        self._last_activity_time = time.time()
        await self.push_frame(frame, direction)

    async def _handle_vad_stopped(self, frame: VADUserStoppedSpeakingFrame, direction: FrameDirection):
        """Handle VAD user stopped speaking."""
        self._vad_user_speaking = False
        self._last_activity_time = time.time()

        if self.use_vad_boundary and self._text_buffer.strip():
            await self._commit_final(self._text_buffer, "VAD")
        else:
            self._clear_committed_prefix("VAD_no_text")
            self._reset_buffer()

        await self.push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        # Start timeout monitor on first frame
        if not self._running:
            self._running = True
            if self.silence_timeout_secs > 0:
                self._timeout_task = asyncio.create_task(self._silence_timeout_monitor())

        # Handle EndFrame/EndTaskFrame - flush buffer before shutdown
        if isinstance(frame, (EndFrame, EndTaskFrame)):
            await self._flush_on_end()
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_transcription(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_started(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_stopped(frame, direction)
        else:
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    async def _flush_on_end(self):
        """Flush remaining buffer content on stream end."""
        if self._text_buffer.strip():
            clean_text = self._strip_special_tokens(self._text_buffer).strip()
            if clean_text and not self._is_duplicate_commit(clean_text):
                self._last_committed_text = clean_text
                self._last_commit_time = time.time()
                self._utterance_count += 1

                if self._display:
                    self._display.show_final(clean_text, "stream_end")
                else:
                    logger.info(f"[FINAL #{self._utterance_count}] (stream_end) {clean_text}")

                frame = TranscriptionFrame(
                    text=clean_text,
                    user_id="",
                    timestamp=time_now_iso8601(),
                    language=self.language,
                    result={"text": clean_text, "is_final": True, "reason": "stream_end"},
                )
                await self.push_frame(frame)

        self._reset_buffer()
        self._committed_prefix = ""
        self._committed_prefix_timestamp = 0.0

    def reset(self):
        """Reset aggregator state."""
        self._text_buffer = ""
        self._last_displayed_text = ""
        self._last_activity_time = time.time()
        self._last_committed_text = ""
        self._last_commit_time = 0.0
        self._committed_prefix = ""
        self._committed_prefix_timestamp = 0.0
        if self._display:
            self._display.reset()

    async def cleanup(self):
        """Clean up resources."""
        await self._flush_on_end()

        # Show session summary
        if self._display:
            self._display.show_summary()

        self._running = False
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None
