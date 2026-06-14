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
Speaker-Aware STT Text Aggregator.

This service combines STT transcription with speaker diarization results
to provide multi-speaker transcription with automatic speaker change detection.

Features:
- Speaker change detection and automatic line breaks
- VAD-based utterance boundary detection
- Punctuation-based sentence boundary detection
- Real-time console display with speaker colors
- Up to 4 speakers support
"""

import asyncio
import sys
import time
from typing import Optional, Set, List, Dict
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

from nemo.agents.voice_agent.pipecat.frames.frames import (
    DiarResultFrame,
    SpeakerTranscriptionFrame,
    SpeakerStatusFrame,
    SpeakerInfo,
)


class SpeakerConsoleDisplay:
    """
    Real-time console display with speaker colors and ASCII-safe indicators.

    Features:
    - Color-coded output per speaker (up to 4 speakers)
    - Overwrites current line for interim transcriptions
    - Commits line with newline for final transcriptions
    - ASCII-safe characters for Linux terminal compatibility
    """

    # ANSI codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GRAY = "\033[90m"

    # Speaker colors (Blue, Green, Yellow, Magenta)
    SPEAKER_COLORS = [
        "\033[94m",  # Blue - Speaker 0
        "\033[92m",  # Green - Speaker 1
        "\033[93m",  # Yellow - Speaker 2
        "\033[95m",  # Magenta - Speaker 3
    ]

    # ASCII-safe speaker indicators (no emoji - works on all terminals)
    SPEAKER_INDICATORS = ["[1]", "[2]", "[3]", "[4]"]
    SPEAKER_DOTS = ["*", "+", "#", "@"]  # For compact display

    # Status colors
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    def __init__(self, use_colors: bool = True, max_speakers: int = 4):
        self.use_colors = use_colors
        self.max_speakers = min(max_speakers, 4)
        self._last_line_length = 0
        self._utterance_count = 0
        self._last_speaker_id: Optional[int] = None

    def _clear_line(self):
        """Clear current line."""
        sys.stdout.write("\r" + " " * self._last_line_length + "\r")
        sys.stdout.flush()

    def _get_speaker_color(self, speaker_id: Optional[int]) -> str:
        """Get ANSI color code for speaker."""
        if speaker_id is None or not self.use_colors:
            return ""
        return self.SPEAKER_COLORS[speaker_id % len(self.SPEAKER_COLORS)]

    def _get_speaker_indicator(self, speaker_id: Optional[int]) -> str:
        """Get ASCII-safe indicator for speaker."""
        if speaker_id is None:
            return "[?]"
        return self.SPEAKER_INDICATORS[speaker_id % len(self.SPEAKER_INDICATORS)]

    def _get_speaker_dot(self, speaker_id: Optional[int]) -> str:
        """Get compact ASCII dot for speaker."""
        if speaker_id is None:
            return "?"
        return self.SPEAKER_DOTS[speaker_id % len(self.SPEAKER_DOTS)]

    def _get_speaker_label(self, speaker_id: Optional[int]) -> str:
        """Get speaker label string."""
        if speaker_id is None:
            return "[???]"
        return f"[S{speaker_id + 1}]"

    def show_interim(self, text: str, speaker_id: Optional[int] = None):
        """Display interim text (overwrites current line)."""
        if not text:
            return

        color = self._get_speaker_color(speaker_id)
        speaker_label = self._get_speaker_label(speaker_id)

        if self.use_colors:
            display = f"\r{self.CYAN}>{self.RESET} {color}{speaker_label}{self.RESET} {self.DIM}{text}{self.RESET}"
        else:
            display = f"\r> {speaker_label} {text}"

        # Pad to ensure clean overwrite
        visible_len = len(text) + 15
        if visible_len < self._last_line_length:
            display += " " * (self._last_line_length - visible_len)
        self._last_line_length = max(visible_len, self._last_line_length)

        sys.stdout.write(display)
        sys.stdout.flush()

    def show_final(self, text: str, speaker_id: Optional[int] = None, reason: str = ""):
        """Display final text with speaker info (commits with newline)."""
        if not text:
            self._clear_line()
            self._last_line_length = 0
            return

        self._utterance_count += 1
        self._clear_line()

        color = self._get_speaker_color(speaker_id)
        speaker_label = self._get_speaker_label(speaker_id)

        # Build reason tag (compact)
        reason_tag = ""
        if reason:
            reason_abbrev = {"speaker_change": "SPK", "VAD": "VAD", "EOU": "EOU", "boundary": "BND", "timeout": "TMO"}
            abbrev = reason_abbrev.get(reason, reason[:3].upper())
            if self.use_colors:
                reason_tag = f" {self.GRAY}[{abbrev}]{self.RESET}"
            else:
                reason_tag = f" [{abbrev}]"

        # Build final display - clean and premium
        if self.use_colors:
            line = (
                f"{self.GREEN}#{self._utterance_count:03d}{self.RESET} "
                f"{color}{speaker_label}{self.RESET} "
                f"{self.WHITE}{text}{self.RESET}{reason_tag}"
            )
        else:
            line = f"#{self._utterance_count:03d} {speaker_label} {text}{reason_tag}"

        print(line)
        sys.stdout.flush()
        self._last_line_length = 0
        self._last_speaker_id = speaker_id

    def show_speaker_change(self, old_speaker: Optional[int], new_speaker: Optional[int]):
        """Display speaker change notification (compact format)."""
        if not self.use_colors:
            return

        old_label = f"S{old_speaker + 1}" if old_speaker is not None else "?"
        new_label = f"S{new_speaker + 1}" if new_speaker is not None else "?"
        old_color = self._get_speaker_color(old_speaker) if old_speaker is not None else self.GRAY
        new_color = self._get_speaker_color(new_speaker) if new_speaker is not None else self.GRAY

        line = f"     {self.GRAY}|{self.RESET} {old_color}{old_label}{self.RESET} {self.GRAY}->{self.RESET} {new_color}{new_label}{self.RESET}"
        print(line)
        sys.stdout.flush()

    def show_timeout_flush(self, text: str, speaker_id: Optional[int] = None):
        """Display timeout-triggered flush."""
        if not text:
            self._clear_line()
            self._last_line_length = 0
            return

        self._utterance_count += 1
        self._clear_line()

        color = self._get_speaker_color(speaker_id)
        speaker_label = self._get_speaker_label(speaker_id)

        if self.use_colors:
            line = (
                f"{self.YELLOW}#{self._utterance_count:03d}{self.RESET} "
                f"{color}{speaker_label}{self.RESET} "
                f"{self.WHITE}{text}{self.RESET} {self.GRAY}[TMO]{self.RESET}"
            )
        else:
            line = f"#{self._utterance_count:03d} {speaker_label} {text} [TMO]"

        print(line)
        sys.stdout.flush()
        self._last_line_length = 0

    def reset(self):
        """Reset display state."""
        self._clear_line()
        self._last_line_length = 0


class SpeakerAwareAggregator(FrameProcessor):
    """
    Text aggregator with speaker change detection.

    Combines STT transcription with speaker diarization to provide:
    - Automatic line breaks on speaker change
    - Speaker-attributed transcriptions
    - Real-time speaker status updates

    Commit triggers (in priority order):
    1. Speaker change (highest priority)
    2. VAD stop
    3. Punctuation boundary
    4. Max words
    5. Silence timeout
    """

    def __init__(
        self,
        # Speaker settings
        max_speakers: int = 4,
        use_speaker_boundary: bool = True,
        min_words_for_speaker_commit: int = 1,
        # Boundary detection settings
        eou_string: str = "<EOU>",
        eob_string: str = "<EOB>",
        sentence_end_punctuation: str = ".?!。？！",
        use_vad_boundary: bool = True,
        use_punctuation_boundary: bool = True,
        min_words_for_punctuation_boundary: int = 3,
        max_words_before_commit: int = 20,
        # Timeout settings
        silence_timeout_secs: float = 3.0,
        silence_check_interval_secs: float = 0.5,
        # Display settings
        language: Language = Language.EN_US,
        use_realtime_display: bool = True,
        use_colors: bool = True,
        **kwargs,
    ):
        """
        Initialize the speaker-aware aggregator.

        Args:
            max_speakers: Maximum number of speakers to track (1-4)
            use_speaker_boundary: Trigger commit on speaker change
            min_words_for_speaker_commit: Minimum words before speaker change commits
            eou_string: End of utterance token string
            eob_string: End of backchannel token string
            sentence_end_punctuation: Characters that indicate sentence end
            use_vad_boundary: Use VAD stop as sentence boundary
            use_punctuation_boundary: Use punctuation as sentence boundary
            min_words_for_punctuation_boundary: Minimum words before punctuation triggers
            max_words_before_commit: Maximum words before forcing a commit
            silence_timeout_secs: Seconds of silence before flushing buffer
            silence_check_interval_secs: Interval for checking silence timeout
            language: Language for transcription frames
            use_realtime_display: Use real-time console display
            use_colors: Use ANSI colors in console output
        """
        super().__init__(**kwargs)

        # Speaker settings
        self.max_speakers = min(max_speakers, 4)
        self.use_speaker_boundary = use_speaker_boundary
        self.min_words_for_speaker_commit = min_words_for_speaker_commit

        # Boundary settings
        self.eou_string = eou_string
        self.eob_string = eob_string
        self.sentence_end_punctuation: Set[str] = set(sentence_end_punctuation)
        self.use_vad_boundary = use_vad_boundary
        self.use_punctuation_boundary = use_punctuation_boundary
        self.min_words_for_punctuation_boundary = min_words_for_punctuation_boundary
        self.max_words_before_commit = max_words_before_commit

        # Timeout settings
        self.silence_timeout_secs = silence_timeout_secs
        self.silence_check_interval_secs = silence_check_interval_secs
        self.language = language

        # Internal state
        self._text_buffer: str = ""
        self._last_displayed_text: str = ""
        self._vad_user_speaking: bool = False
        self._utterance_count: int = 0

        # Speaker tracking
        self._current_speaker_id: Optional[int] = None
        self._pending_speaker_id: Optional[int] = None  # Speaker from diar, not yet applied
        self._speaker_status: Dict[int, SpeakerInfo] = {
            i: SpeakerInfo(id=i, active=False)
            for i in range(self.max_speakers)
        }

        # Timing
        self._last_activity_time: float = time.time()
        self._timeout_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Display
        self._display = (
            SpeakerConsoleDisplay(use_colors=use_colors, max_speakers=max_speakers)
            if use_realtime_display
            else None
        )

        logger.info(f"[Speaker-Aggregator] Initialized:")
        logger.info(f"[Speaker-Aggregator]   Max speakers: {max_speakers}")
        logger.info(f"[Speaker-Aggregator]   Speaker boundary: {use_speaker_boundary}")
        logger.info(f"[Speaker-Aggregator]   VAD boundary: {use_vad_boundary}")
        logger.info(f"[Speaker-Aggregator]   Punctuation boundary: {use_punctuation_boundary}")
        logger.info(f"[Speaker-Aggregator]   Max words: {max_words_before_commit}")
        logger.info(f"[Speaker-Aggregator]   Silence timeout: {silence_timeout_secs}s")

    async def _silence_timeout_monitor(self):
        """Background task to monitor silence and flush buffer on timeout."""
        logger.debug("[Speaker-Aggregator] Silence timeout monitor started")

        while self._running:
            try:
                await asyncio.sleep(self.silence_check_interval_secs)

                if self.silence_timeout_secs <= 0 or self._vad_user_speaking:
                    continue

                if self._text_buffer.strip():
                    elapsed = time.time() - self._last_activity_time
                    if elapsed >= self.silence_timeout_secs:
                        logger.debug(f"[Speaker-Aggregator] Silence timeout after {elapsed:.1f}s")
                        await self._commit_final_timeout(self._text_buffer)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Speaker-Aggregator] Error in silence monitor: {e}")

        logger.debug("[Speaker-Aggregator] Silence timeout monitor stopped")

    def _update_activity_time(self):
        """Update the last activity timestamp."""
        self._last_activity_time = time.time()

    def _strip_special_tokens(self, text: str) -> str:
        """Remove EOU/EOB tokens and clean SentencePiece artifacts from text.

        Cleans:
        - EOU/EOB tokens (end of utterance/block markers)
        - SentencePiece token boundary character (▁ -> space)
        - Unicode replacement characters (�)
        - Multiple consecutive spaces
        """
        # Remove EOU/EOB tokens
        text = text.replace(self.eou_string, "").replace(self.eob_string, "")
        # Replace SentencePiece token boundary with space
        text = text.replace("\u2581", " ")  # ▁ -> space
        # Remove Unicode replacement characters
        text = text.replace("\uFFFD", "")  # � (replacement character)
        # Collapse multiple spaces into one
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _detect_eou(self, text: str) -> bool:
        """Check if text contains end of utterance token."""
        return self.eou_string in text or self.eob_string in text

    def _get_word_count(self, text: str) -> int:
        """Get word count from text."""
        clean_text = self._strip_special_tokens(text).strip()
        return len(clean_text.split()) if clean_text else 0

    def _detect_sentence_boundary(self, text: str) -> bool:
        """Check if text has reached a sentence boundary."""
        clean_text = self._strip_special_tokens(text).strip()
        if not clean_text:
            return False

        word_count = len(clean_text.split())

        # Check max word count
        if word_count >= self.max_words_before_commit:
            return True

        # Check punctuation boundary
        if self.use_punctuation_boundary:
            if clean_text[-1] in self.sentence_end_punctuation:
                if word_count >= self.min_words_for_punctuation_boundary:
                    return True

        return False

    async def _update_speaker_status(self, active_speaker_id: Optional[int]):
        """Update and push speaker status frame."""
        # Update status
        for speaker_id in self._speaker_status:
            self._speaker_status[speaker_id].active = (speaker_id == active_speaker_id)
            if speaker_id == active_speaker_id:
                self._speaker_status[speaker_id].activity_level = 100.0
            else:
                # Decay activity level
                self._speaker_status[speaker_id].activity_level = max(
                    0, self._speaker_status[speaker_id].activity_level - 10
                )

        # Create and push status frame
        status_frame = SpeakerStatusFrame(
            speakers=list(self._speaker_status.values()),
            active_speaker_id=active_speaker_id,
            total_speakers=self.max_speakers,
        )
        await self.push_frame(status_frame)

    async def _commit_final(self, text: str, reason: str):
        """Commit final transcription and reset buffer."""
        clean_text = self._strip_special_tokens(text).strip()

        if not clean_text:
            if self._display:
                self._display.reset()
            self._text_buffer = ""
            self._last_displayed_text = ""
            return

        self._utterance_count += 1

        # Display final text with speaker info
        if self._display:
            self._display.show_final(clean_text, self._current_speaker_id, reason)
        else:
            speaker_label = f"[SPK{self._current_speaker_id}]" if self._current_speaker_id is not None else "[???]"
            logger.info(f"[FINAL #{self._utterance_count}] {speaker_label} ({reason}) {clean_text}")

        # Emit speaker transcription frame
        frame = SpeakerTranscriptionFrame(
            text=clean_text,
            speaker_id=self._current_speaker_id,
            is_final=True,
            reason=reason,
            timestamp=time_now_iso8601(),
            language=self.language,
        )
        await self.push_frame(frame)

        # Also emit standard TranscriptionFrame for backward compatibility
        compat_frame = TranscriptionFrame(
            text=clean_text,
            user_id="",
            timestamp=time_now_iso8601(),
            language=self.language,
            result={
                "text": clean_text,
                "is_final": True,
                "reason": reason,
                "speaker_id": self._current_speaker_id,
            },
        )
        await self.push_frame(compat_frame)

        # Reset buffer
        self._text_buffer = ""
        self._last_displayed_text = ""

    async def _commit_final_timeout(self, text: str):
        """Commit final transcription due to silence timeout."""
        clean_text = self._strip_special_tokens(text).strip()

        if not clean_text:
            if self._display:
                self._display.reset()
            self._text_buffer = ""
            self._last_displayed_text = ""
            return

        self._utterance_count += 1

        if self._display:
            self._display.show_timeout_flush(clean_text, self._current_speaker_id)
        else:
            speaker_label = f"[SPK{self._current_speaker_id}]" if self._current_speaker_id is not None else "[???]"
            logger.info(f"[TIMEOUT #{self._utterance_count}] {speaker_label} {clean_text}")

        # Emit frames
        frame = SpeakerTranscriptionFrame(
            text=clean_text,
            speaker_id=self._current_speaker_id,
            is_final=True,
            reason="timeout",
            timestamp=time_now_iso8601(),
            language=self.language,
        )
        await self.push_frame(frame)

        compat_frame = TranscriptionFrame(
            text=clean_text,
            user_id="",
            timestamp=time_now_iso8601(),
            language=self.language,
            result={
                "text": clean_text,
                "is_final": True,
                "reason": "timeout",
                "speaker_id": self._current_speaker_id,
            },
        )
        await self.push_frame(compat_frame)

        self._text_buffer = ""
        self._last_displayed_text = ""

    async def _update_interim(self, text: str):
        """Update interim display."""
        clean_text = self._strip_special_tokens(text).strip()

        if not clean_text:
            return

        if clean_text == self._last_displayed_text:
            return

        self._last_displayed_text = clean_text

        if self._display:
            self._display.show_interim(clean_text, self._current_speaker_id)

        # Emit interim frame
        frame = InterimTranscriptionFrame(
            text=clean_text,
            user_id="",
            timestamp=time_now_iso8601(),
            language=self.language,
            result={
                "text": clean_text,
                "is_final": False,
                "speaker_id": self._current_speaker_id,
            },
        )
        await self.push_frame(frame)

    async def _handle_diar_result(self, frame: DiarResultFrame, direction: FrameDirection):
        """Handle diarization result frame."""
        new_speaker_id = frame.diar_result if isinstance(frame.diar_result, int) else None

        if new_speaker_id is None:
            return

        # Check for speaker change
        if self.use_speaker_boundary and new_speaker_id != self._current_speaker_id:
            old_speaker = self._current_speaker_id

            # Commit current buffer if it has content
            if self._text_buffer.strip():
                word_count = self._get_word_count(self._text_buffer)
                if word_count >= self.min_words_for_speaker_commit:
                    await self._commit_final(self._text_buffer, "speaker_change")

                    # Show speaker change notification
                    if self._display:
                        self._display.show_speaker_change(old_speaker, new_speaker_id)

            # Update current speaker
            self._current_speaker_id = new_speaker_id
            logger.debug(f"[Speaker-Aggregator] Speaker changed: {old_speaker} → {new_speaker_id}")

            # Update speaker status
            await self._update_speaker_status(new_speaker_id)

        elif self._current_speaker_id is None:
            # First speaker detection
            self._current_speaker_id = new_speaker_id
            await self._update_speaker_status(new_speaker_id)

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame, direction: FrameDirection
    ):
        """Handle incoming transcription frames."""
        incoming_text = frame.text

        self._update_activity_time()

        # Check for EOU/EOB tokens (highest priority)
        if self._detect_eou(incoming_text):
            await self._commit_final(incoming_text, "EOU")
            return

        # Update buffer
        self._text_buffer = incoming_text

        # Check for sentence boundary
        if self._detect_sentence_boundary(incoming_text):
            await self._commit_final(incoming_text, "boundary")
            return

        # Update interim display
        await self._update_interim(incoming_text)

    async def _handle_vad_started(self, frame: VADUserStartedSpeakingFrame, direction: FrameDirection):
        """Handle VAD user started speaking."""
        self._vad_user_speaking = True
        self._update_activity_time()
        await self.push_frame(frame, direction)

    async def _handle_vad_stopped(self, frame: VADUserStoppedSpeakingFrame, direction: FrameDirection):
        """Handle VAD user stopped speaking."""
        self._vad_user_speaking = False
        self._update_activity_time()

        # Commit any buffered text when VAD stops
        if self.use_vad_boundary and self._text_buffer.strip():
            await self._commit_final(self._text_buffer, "VAD")
        elif self._display:
            self._display.reset()
            self._text_buffer = ""
            self._last_displayed_text = ""

        # Update speaker status (no active speaker)
        await self._update_speaker_status(None)

        await self.push_frame(frame, direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        # Start timeout monitor on first frame
        if not self._running:
            self._running = True
            if self.silence_timeout_secs > 0:
                self._timeout_task = asyncio.create_task(self._silence_timeout_monitor())

        if isinstance(frame, DiarResultFrame):
            await self._handle_diar_result(frame, direction)
        elif isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            await self._handle_transcription(frame, direction)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._handle_vad_started(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._handle_vad_stopped(frame, direction)
        else:
            await self.push_frame(frame, direction)

    def reset(self):
        """Reset aggregator state."""
        self._text_buffer = ""
        self._last_displayed_text = ""
        self._current_speaker_id = None
        self._last_activity_time = time.time()
        if self._display:
            self._display.reset()

    async def cleanup(self):
        """Clean up resources."""
        self._running = False
        if self._timeout_task:
            self._timeout_task.cancel()
            try:
                await self._timeout_task
            except asyncio.CancelledError:
                pass
            self._timeout_task = None
