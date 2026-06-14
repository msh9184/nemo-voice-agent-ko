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

from typing import Optional, List, Dict, Any
import asyncio
import re
import time
import uuid
import struct

from loguru import logger
from pydantic import BaseModel, Field
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIBotLLMTextMessage,
    RTVIBotTranscriptionMessage,
    RTVIMessage,
)
from pipecat.processors.frameworks.rtvi import RTVIObserver as _RTVIObserver
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVITextMessageData

from nemo.agents.voice_agent.pipecat.frames.frames import SpeakerStatusFrame, SpeakerTranscriptionFrame
from nemo.agents.voice_agent.pipecat.utils.text.simple_text_aggregator import SimpleSegmentedTextAggregator


def _generate_id() -> str:
    """Generate a unique message ID."""
    return str(uuid.uuid4())


# Custom RTVI message types for speaker diarization
class RTVISpeakerStatusData(BaseModel):
    """Data for speaker status update message."""
    speakers: List[Dict[str, Any]]
    active_speaker_id: Optional[int] = None
    total_speakers: int = 0


class RTVISpeakerTranscriptionData(BaseModel):
    """Data for speaker-attributed transcription message."""
    text: str
    speaker_id: Optional[int] = None
    speaker_name: str = ""
    speaker_color: str = ""
    is_final: bool = False
    reason: str = ""
    timestamp: str = ""


class RTVISpeakerStatusMessage(RTVIMessage):
    """RTVI message for speaker status updates."""
    id: str = Field(default_factory=_generate_id)
    type: str = "speaker-status"
    data: RTVISpeakerStatusData


class RTVISpeakerTranscriptionMessage(RTVIMessage):
    """RTVI message for speaker-attributed transcription."""
    id: str = Field(default_factory=_generate_id)
    type: str = "speaker-transcription"
    data: RTVISpeakerTranscriptionData


# New message types for LLM streaming and TTS sync
class RTVIBotLLMStreamData(BaseModel):
    """Data for streaming LLM output with timing info."""
    text: str  # Current text chunk
    accumulated: str  # Full accumulated text so far
    is_sentence_end: bool = False  # Whether this completes a sentence
    timestamp: float = 0.0


class RTVIBotLLMStreamMessage(RTVIMessage):
    """RTVI message for streaming LLM output."""
    id: str = Field(default_factory=_generate_id)
    type: str = "bot-llm-stream"
    data: RTVIBotLLMStreamData


class RTVIBotTTSTextData(BaseModel):
    """Data for text being sent to TTS."""
    text: str  # Text chunk being synthesized
    chunk_id: int = 0  # Sequential chunk ID for ordering
    is_final: bool = False  # Whether this is the last chunk


class RTVIBotTTSTextMessage(RTVIMessage):
    """RTVI message for text being sent to TTS (for sync)."""
    id: str = Field(default_factory=_generate_id)
    type: str = "bot-tts-text"
    data: RTVIBotTTSTextData


class RTVIBotStatusData(BaseModel):
    """Data for bot status updates."""
    status: str  # "thinking", "speaking", "idle"
    tts_playing: bool = False
    current_text: str = ""


class RTVIBotStatusMessage(RTVIMessage):
    """RTVI message for bot status updates."""
    id: str = Field(default_factory=_generate_id)
    type: str = "bot-status"
    data: RTVIBotStatusData


class RTVIServerConfigData(BaseModel):
    """Data for server configuration message."""
    stt: Optional[Dict[str, Any]] = None
    llm: Optional[Dict[str, Any]] = None
    tts: Optional[Dict[str, Any]] = None
    diar: Optional[Dict[str, Any]] = None


class RTVIServerConfigMessage(RTVIMessage):
    """RTVI message for server configuration (model info)."""
    id: str = Field(default_factory=_generate_id)
    type: str = "server-config"
    data: RTVIServerConfigData


class RTVIBotAudioLevelData(BaseModel):
    """Data for bot audio level (TTS output amplitude).

    Used for real-time audio visualization on the client side.
    Sent ~30-60 times per second during TTS playback.
    """
    level: float = 0.0  # Normalized audio level (0.0 to 1.0)
    peak: float = 0.0   # Peak level for visualization effects
    is_speaking: bool = False  # Whether bot is currently speaking


class RTVIBotAudioLevelMessage(RTVIMessage):
    """RTVI message for bot audio level (for visualization)."""
    id: str = Field(default_factory=_generate_id)
    type: str = "bot-audio-level"
    data: RTVIBotAudioLevelData


class SpeakerFrameProcessor(FrameProcessor):
    """
    Frame processor that converts SpeakerStatusFrame and SpeakerTranscriptionFrame
    to RTVI messages and sends them to the client.

    This processor should be placed in the pipeline after SpeakerAwareAggregator
    and before or alongside RTVIProcessor.

    NOTE: Uses RTVIObserver for message transport (push_transport_message_urgent)
    because RTVIProcessor._push_transport_message may not work correctly.
    """

    def __init__(self, rtvi_observer: 'RTVIObserver', **kwargs):
        super().__init__(**kwargs)
        self._rtvi_observer = rtvi_observer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and convert speaker frames to RTVI messages."""
        await super().process_frame(frame, direction)

        if isinstance(frame, SpeakerStatusFrame):
            await self._handle_speaker_status(frame)
            # Also push for potential downstream processors
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, SpeakerTranscriptionFrame):
            await self._handle_speaker_transcription(frame)
            # Also push the frame downstream for other processors
            await self.push_frame(frame, direction)
            return

        # Pass all other frames through
        await self.push_frame(frame, direction)

    async def _handle_speaker_status(self, frame: SpeakerStatusFrame):
        """Convert SpeakerStatusFrame to RTVI message and send."""
        speakers_data = [
            {
                "id": s.id,
                "active": s.active,
                "name": s.name,
                "color": s.color,
                "activity_level": s.activity_level,
            }
            for s in frame.speakers
        ]

        message = RTVISpeakerStatusMessage(
            data=RTVISpeakerStatusData(
                speakers=speakers_data,
                active_speaker_id=frame.active_speaker_id,
                total_speakers=frame.total_speakers,
            )
        )

        if frame.active_speaker_id is not None:
            logger.info(f"RTVI | speaker-status -> S{frame.active_speaker_id + 1}")

        # Use RTVIObserver's push_transport_message_urgent for reliable message delivery
        await self._rtvi_observer.push_transport_message_urgent(message)

    async def _handle_speaker_transcription(self, frame: SpeakerTranscriptionFrame):
        """Convert SpeakerTranscriptionFrame to RTVI message and send."""
        speaker_name = f"Speaker {frame.speaker_id + 1}" if frame.speaker_id is not None else "Unknown"
        colors = ["#58a6ff", "#3fb950", "#d29922", "#a371f7"]
        speaker_color = colors[frame.speaker_id % len(colors)] if frame.speaker_id is not None else "#888888"

        message = RTVISpeakerTranscriptionMessage(
            data=RTVISpeakerTranscriptionData(
                text=frame.text,
                speaker_id=frame.speaker_id,
                speaker_name=speaker_name,
                speaker_color=speaker_color,
                is_final=frame.is_final,
                reason=frame.reason,
                timestamp=frame.timestamp,
            )
        )

        if frame.is_final:
            logger.info(f"RTVI | speaker-transcription -> [{speaker_name}] {frame.text[:50]}...")
        else:
            logger.debug(f"RTVI | speaker-partial -> [{speaker_name}] {frame.text[:30]}...")

        # Use RTVIObserver's push_transport_message_urgent for reliable message delivery
        await self._rtvi_observer.push_transport_message_urgent(message)


class RTVIObserver(_RTVIObserver):
    def __init__(
        self, rtvi: RTVIProcessor, text_aggregator: Optional[SimpleSegmentedTextAggregator] = None, *args, **kwargs
    ):
        super().__init__(rtvi, *args, **kwargs)
        self._text_aggregator = text_aggregator if text_aggregator else SimpleSegmentedTextAggregator("?!:.")

        # LLM streaming state
        self._llm_accumulated_text = ""  # Full accumulated LLM output
        self._llm_start_time = 0.0  # When current LLM response started (for timestamps)
        self._llm_last_frame_time = 0.0  # When last LLM frame was received (for new response detection)
        self._tts_chunk_id = 0  # Sequential TTS chunk counter
        self._bot_status = "idle"  # idle, thinking, speaking
        self._tts_playing = False
        self._current_tts_text = ""

        # Audio level tracking for visualization
        self._audio_level_last_send = 0.0  # Last time audio level was sent
        self._audio_level_interval = 0.033  # ~30fps throttle (33ms)
        self._audio_level_peak = 0.0  # Peak level for smooth decay
        self._audio_level_peak_decay = 0.95  # Peak decay factor

        # TTS enabled flag - set via set_tts_enabled() from server config
        # When TTS is disabled (LLM-Only mode), we need to manually send 'idle'
        # status when LLM response completes
        self._tts_enabled = True  # Default to True for backward compatibility

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace in LLM output for clean display.

        Some LLM models output tokens with excessive spaces between words,
        including various Unicode whitespace characters (NBSP, etc.).
        This function:
        1. Collapses all horizontal whitespace to single space
        2. Normalizes multiple blank lines to single blank line
        3. Cleans up spaces around newlines

        Args:
            text: Raw LLM output text

        Returns:
            Normalized text with proper spacing and line breaks
        """
        if not text:
            return text

        # Step 1: Collapse ALL horizontal whitespace (space, tab, NBSP, etc.) to single space
        # [^\S\n\r] matches any whitespace character EXCEPT newline/carriage return
        normalized = re.sub(r'[^\S\n\r]+', ' ', text)

        # Step 2: Clean up spaces around newlines
        # Remove trailing spaces before newline
        normalized = re.sub(r' +\n', '\n', normalized)
        # Remove leading spaces after newline (except for intentional indentation)
        normalized = re.sub(r'\n +', '\n', normalized)

        # Step 3: Normalize multiple consecutive newlines to at most 2 (one blank line)
        # This preserves paragraph breaks while preventing excessive blank lines
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)

        # Step 4: Clean up lines - strip each line and rejoin
        # This catches any remaining whitespace issues
        lines = normalized.split('\n')
        cleaned_lines = []
        for line in lines:
            # Strip whitespace from each line
            stripped = line.strip()
            # Collapse any remaining multiple spaces within the line
            stripped = re.sub(r' {2,}', ' ', stripped)
            cleaned_lines.append(stripped)

        normalized = '\n'.join(cleaned_lines)

        return normalized

    async def _handle_llm_text_frame(self, frame: LLMTextFrame):
        """Handle LLM text output frames with enhanced streaming."""
        # CRITICAL FIX: Reset accumulated text when starting a new LLM response
        #
        # Problem: BotStartedSpeakingFrame and BotStoppedSpeakingFrame may not reach
        # RTVIObserver (they're generated by TTS/transport layer, not the main pipeline).
        # This means _bot_status can stay "thinking" even after TTS finishes.
        #
        # Solution: Use BOTH status check AND time-based detection:
        # 1. Status check: Reset if status is "idle" or "speaking"
        # 2. Time check: If status is "thinking" but >2s since last frame, treat as new response
        #    (This catches the case where status is stuck at "thinking")
        #
        current_time = time.time()
        time_since_last_frame = current_time - self._llm_last_frame_time if self._llm_last_frame_time > 0 else 999.0

        # Detect new response using multiple signals
        is_new_response_by_status = self._bot_status in ("idle", "speaking")
        is_new_response_by_time = (
            self._bot_status == "thinking" and
            time_since_last_frame > 2.0 and  # >2 seconds gap indicates new response
            self._llm_accumulated_text  # Only if there was previous text (not first frame ever)
        )

        if is_new_response_by_status or is_new_response_by_time:
            # This is the start of a new LLM response
            if self._llm_accumulated_text:
                reason = "status change" if is_new_response_by_status else f"time gap ({time_since_last_frame:.1f}s)"
                logger.info(f"Resetting LLM state: was '{self._bot_status}', reason: {reason}, had {len(self._llm_accumulated_text)} chars")
            self._llm_accumulated_text = ""
            self._tts_chunk_id = 0
            self._current_tts_text = ""
            await self._text_aggregator.reset()
            self._llm_start_time = current_time
            await self._update_bot_status("thinking")
        elif not self._llm_accumulated_text:
            # Edge case: status is already "thinking" but no accumulated text yet (first frame)
            self._llm_start_time = current_time

        # Update last frame time for new response detection
        self._llm_last_frame_time = current_time

        # Accumulate text
        self._llm_accumulated_text += frame.text

        # Normalize accumulated text for display (collapse multiple spaces)
        normalized_accumulated = self._normalize_text(self._llm_accumulated_text)

        # Send original bot-llm-text for compatibility
        message = RTVIBotLLMTextMessage(data=RTVITextMessageData(text=frame.text))
        await self.push_transport_message_urgent(message)

        # Check for sentence completion
        completed_text = await self._text_aggregator.aggregate(frame.text)
        is_sentence_end = completed_text is not None

        # Send enhanced streaming message with NORMALIZED accumulated text
        stream_message = RTVIBotLLMStreamMessage(
            data=RTVIBotLLMStreamData(
                text=frame.text,
                accumulated=normalized_accumulated,
                is_sentence_end=is_sentence_end,
                timestamp=time.time() - self._llm_start_time,
            )
        )
        await self.push_transport_message_urgent(stream_message)

        # If sentence completed, send TTS text message
        if completed_text:
            await self._push_tts_text(completed_text)
            await self._push_bot_transcription(completed_text)

    async def _push_tts_text(self, text: str):
        """Push text chunk being sent to TTS for client sync."""
        # Normalize whitespace before sending
        normalized_text = self._normalize_text(text)

        if len(normalized_text.strip()) > 0:
            self._tts_chunk_id += 1
            self._current_tts_text = normalized_text

            message = RTVIBotTTSTextMessage(
                data=RTVIBotTTSTextData(
                    text=normalized_text,
                    chunk_id=self._tts_chunk_id,
                    is_final=False,  # Will be set true on LLM completion
                )
            )
            logger.info(f"TTS text chunk #{self._tts_chunk_id}: `{normalized_text[:50]}...`")
            await self.push_transport_message_urgent(message)

    async def _push_bot_transcription(self, text: str):
        """Push accumulated bot transcription as a message."""
        if len(text.strip()) > 0:
            message = RTVIBotTranscriptionMessage(data=RTVITextMessageData(text=text))
            logger.debug(f"Pushing bot transcription: `{text}`")
            await self.push_transport_message_urgent(message)
            self._bot_transcription = ""

    async def _update_bot_status(self, status: str):
        """Update and broadcast bot status."""
        self._bot_status = status
        message = RTVIBotStatusMessage(
            data=RTVIBotStatusData(
                status=status,
                tts_playing=self._tts_playing,
                current_text=self._current_tts_text,
            )
        )
        logger.debug(f"Bot status: {status}, TTS playing: {self._tts_playing}")
        await self.push_transport_message_urgent(message)

    async def reset_llm_state(self):
        """Reset LLM accumulated state for new text input.

        This method must be called before processing new text input
        (especially in LLM-Only mode where BotStoppedSpeakingFrame never fires).
        Without this reset, LLM responses would accumulate across messages.

        Note: This method is async because _text_aggregator.reset() is async
        (required by pipecat's TTSService which calls `await reset()`).
        """
        self._llm_accumulated_text = ""
        self._llm_last_frame_time = 0.0  # Reset for new response detection
        self._tts_chunk_id = 0
        self._current_tts_text = ""
        await self._text_aggregator.reset()
        logger.debug("LLM state reset for new text input")

    def set_tts_enabled(self, enabled: bool):
        """Set whether TTS is enabled.

        In LLM-Only mode (TTS disabled), we need to send 'idle' status
        when LLM response completes since BotStoppedSpeakingFrame won't fire.

        Args:
            enabled: True if TTS is enabled, False for LLM-Only mode
        """
        self._tts_enabled = enabled
        logger.info(f"RTVIObserver TTS enabled: {enabled}")

    async def _handle_llm_response_end(self):
        """Handle LLM response completion.

        In LLM-Only mode (no TTS), we need to manually set bot status to 'idle'
        and reset LLM state since BotStoppedSpeakingFrame will never fire.
        """
        if not self._tts_enabled:
            # LLM-Only mode: No TTS to play, so finalize the response now
            logger.debug("LLM response ended (LLM-Only mode), setting status to idle")
            self._llm_accumulated_text = ""
            self._llm_last_frame_time = 0.0  # Reset for new response detection
            self._tts_chunk_id = 0
            self._current_tts_text = ""
            await self._update_bot_status("idle")
        else:
            # Normal mode: TTS will play, BotStoppedSpeakingFrame will handle 'idle'
            logger.debug("LLM response ended, waiting for TTS to finish")

    async def send_server_config(self, config: Dict[str, Any]):
        """Send server configuration (model info) to client.

        Args:
            config: Dictionary containing stt, llm, tts, diar configuration
        """
        message = RTVIServerConfigMessage(
            data=RTVIServerConfigData(
                stt=config.get("stt"),
                llm=config.get("llm"),
                tts=config.get("tts"),
                diar=config.get("diar"),
            )
        )
        logger.info(f"Sending server config to client")
        await self.push_transport_message_urgent(message)

    async def _handle_user_started_speaking(self, frame: Frame):
        """Handle user starting to speak - reset LLM state for new conversation turn.

        This is CRITICAL for voice input mode to prevent LLM response accumulation
        across conversation turns. Without this reset, `_llm_accumulated_text` would
        carry over from the previous bot response, causing responses to concatenate.

        This fix addresses the issue where:
        - Turn 1: Bot says "안녕하세요! 저는 AI 어시스턴트..."
        - Turn 2: User says "오늘 날씨 알려줘"
        - Without reset: Bot response would be "안녕하세요!...현재 위치를..."
        - With reset: Bot response is correctly "현재 위치를..."
        """
        # Reset LLM accumulated state for new conversation turn
        self._llm_accumulated_text = ""
        self._llm_last_frame_time = 0.0  # Reset for new response detection
        self._tts_chunk_id = 0
        self._current_tts_text = ""
        await self._text_aggregator.reset()
        logger.debug("LLM state reset for new user turn (UserStartedSpeakingFrame)")

    async def _handle_bot_speaking_frames(self, frame: Frame):
        """Handle bot speaking state changes."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._tts_playing = True
            await self._update_bot_status("speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._tts_playing = False
            # Reset LLM state when bot finishes speaking
            self._llm_accumulated_text = ""
            self._llm_last_frame_time = 0.0  # Reset for new response detection
            self._tts_chunk_id = 0
            self._current_tts_text = ""
            await self._update_bot_status("idle")

    async def _handle_tts_frames(self, frame: Frame):
        """Handle TTS state changes."""
        if isinstance(frame, TTSStartedFrame):
            logger.debug("TTS started generating audio")
        elif isinstance(frame, TTSStoppedFrame):
            logger.debug("TTS stopped generating audio")

    def _calculate_audio_level(self, audio_bytes: bytes, sample_width: int = 2) -> tuple:
        """Calculate RMS and peak level from audio bytes.

        Args:
            audio_bytes: Raw audio data (int16 PCM format)
            sample_width: Bytes per sample (2 for int16)

        Returns:
            Tuple of (rms_level, peak_level) normalized to 0.0-1.0
        """
        if not audio_bytes or len(audio_bytes) < sample_width:
            return 0.0, 0.0

        try:
            # Parse int16 samples
            num_samples = len(audio_bytes) // sample_width
            if num_samples == 0:
                return 0.0, 0.0

            # Unpack as int16 little-endian
            samples = struct.unpack(f'<{num_samples}h', audio_bytes[:num_samples * sample_width])

            # Calculate RMS (Root Mean Square)
            sum_squares = sum(s * s for s in samples)
            rms = (sum_squares / num_samples) ** 0.5

            # Normalize to 0.0-1.0 (int16 max is 32767)
            rms_normalized = min(1.0, rms / 32767.0)

            # Calculate peak
            peak = max(abs(s) for s in samples) if samples else 0
            peak_normalized = min(1.0, peak / 32767.0)

            # Apply some amplification for better visualization (audio is often quiet)
            rms_normalized = min(1.0, rms_normalized * 2.5)
            peak_normalized = min(1.0, peak_normalized * 1.5)

            return rms_normalized, peak_normalized

        except Exception as e:
            logger.debug(f"Error calculating audio level: {e}")
            return 0.0, 0.0

    async def _handle_tts_audio_frame(self, frame: TTSAudioRawFrame):
        """Handle TTS audio frames and send audio level for visualization.

        Throttled to ~30fps to avoid overwhelming the client.
        """
        current_time = time.time()

        # Throttle audio level messages
        if current_time - self._audio_level_last_send < self._audio_level_interval:
            return

        self._audio_level_last_send = current_time

        # Calculate audio level from the frame
        rms_level, peak_level = self._calculate_audio_level(frame.audio)

        # Update peak with decay for smoother visualization
        self._audio_level_peak = max(
            peak_level,
            self._audio_level_peak * self._audio_level_peak_decay
        )

        # Send audio level message
        message = RTVIBotAudioLevelMessage(
            data=RTVIBotAudioLevelData(
                level=rms_level,
                peak=self._audio_level_peak,
                is_speaking=self._tts_playing,
            )
        )
        await self.push_transport_message_urgent(message)

    async def on_push_frame(self, frame: Frame):
        """Override to handle custom speaker frames and bot status frames.

        NOTE: LLMTextFrame is NOT handled explicitly here. The parent class
        (pipecat's RTVIObserver) handles LLMTextFrame by calling _handle_llm_text_frame,
        which we override to provide enhanced streaming. This avoids double processing.
        """
        # Handle User started speaking - reset LLM state for new conversation turn
        # This is CRITICAL for voice input mode to prevent response accumulation
        if isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            await super().on_push_frame(frame)
            return

        # Handle Bot speaking state changes (custom handling + parent)
        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            await self._handle_bot_speaking_frames(frame)
            await super().on_push_frame(frame)
            return

        # Handle TTS state changes (custom handling + parent)
        if isinstance(frame, (TTSStartedFrame, TTSStoppedFrame)):
            await self._handle_tts_frames(frame)
            await super().on_push_frame(frame)
            return

        # Handle TTS audio frames for visualization (extract audio level, then delegate to parent)
        if isinstance(frame, TTSAudioRawFrame):
            await self._handle_tts_audio_frame(frame)
            await super().on_push_frame(frame)
            return

        # Handle SpeakerStatusFrame (custom only, no parent handling needed)
        if isinstance(frame, SpeakerStatusFrame):
            logger.debug(f"RTVI Observer received SpeakerStatusFrame: active={frame.active_speaker_id}")
            await self._handle_speaker_status_frame(frame)
            return

        # Handle SpeakerTranscriptionFrame (custom only, no parent handling needed)
        if isinstance(frame, SpeakerTranscriptionFrame):
            logger.debug(f"RTVI Observer received SpeakerTranscriptionFrame: {frame.text[:50]}...")
            await self._handle_speaker_transcription_frame(frame)
            return

        # Handle LLM response completion (for LLM-Only mode)
        if isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_response_end()
            await super().on_push_frame(frame)
            return

        # Delegate to parent for all other frames (including LLMTextFrame)
        # Parent's on_push_frame will call our _handle_llm_text_frame for LLMTextFrame
        await super().on_push_frame(frame)

    async def _handle_speaker_status_frame(self, frame: SpeakerStatusFrame):
        """Handle speaker status update frames.

        Sends speaker status to client for UI visualization (activity indicators).
        """
        speakers_data = [
            {
                "id": s.id,
                "active": s.active,
                "name": s.name,
                "color": s.color,
                "activity_level": s.activity_level,
            }
            for s in frame.speakers
        ]

        message = RTVISpeakerStatusMessage(
            data=RTVISpeakerStatusData(
                speakers=speakers_data,
                active_speaker_id=frame.active_speaker_id,
                total_speakers=frame.total_speakers,
            )
        )
        # Log speaker status being sent to client
        if frame.active_speaker_id is not None:
            logger.info(f"RTVI | speaker-status: active=S{frame.active_speaker_id + 1}")
        await self.push_transport_message_urgent(message)

    async def _handle_speaker_transcription_frame(self, frame: SpeakerTranscriptionFrame):
        """Handle speaker-attributed transcription frames.

        Sends transcription with speaker information to client for colored display.
        """
        # Generate speaker name and color
        speaker_name = f"Speaker {frame.speaker_id + 1}" if frame.speaker_id is not None else "Unknown"
        colors = ["#58a6ff", "#3fb950", "#d29922", "#a371f7"]  # Blue, Green, Orange, Purple
        speaker_color = colors[frame.speaker_id % len(colors)] if frame.speaker_id is not None else "#888888"

        message = RTVISpeakerTranscriptionMessage(
            data=RTVISpeakerTranscriptionData(
                text=frame.text,
                speaker_id=frame.speaker_id,
                speaker_name=speaker_name,
                speaker_color=speaker_color,
                is_final=frame.is_final,
                reason=frame.reason,
                timestamp=frame.timestamp,
            )
        )
        logger.debug(f"Pushing speaker transcription: [{speaker_name}] `{frame.text}` (final={frame.is_final})")
        await self.push_transport_message_urgent(message)
