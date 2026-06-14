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
Voice Agent Display Processor.

Intercepts pipeline frames to display clean, real-time console output:
- STT partial/final transcriptions
- LLM streaming responses
- TTS text chunks
"""

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from nemo.agents.voice_agent.pipecat.utils.realtime_display import (
    VoiceAgentDisplay,
    get_voice_agent_display,
)


class VoiceAgentDisplayProcessor(FrameProcessor):
    """
    Frame processor that displays Voice Agent pipeline events to console.

    Provides clean, readable output:
    - STT: Partial (dim, overwrites) and Final (bold, newline)
    - LLM: Accumulated response (displayed on completion)
    - TTS: Text chunks being synthesized

    Add this processor to the pipeline to enable console display.
    """

    def __init__(
        self,
        display: VoiceAgentDisplay = None,
        show_stt: bool = True,
        show_llm: bool = True,
        show_tts: bool = True,
        **kwargs,
    ):
        """
        Initialize the display processor.

        Args:
            display: VoiceAgentDisplay instance (uses global if None)
            show_stt: Show STT transcriptions
            show_llm: Show LLM responses
            show_tts: Show TTS chunks
        """
        super().__init__(**kwargs)

        self._display = display or get_voice_agent_display()
        self._show_stt = show_stt
        self._show_llm = show_llm
        self._show_tts = show_tts

        # Track LLM state
        self._in_llm_response = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and display relevant ones."""
        await super().process_frame(frame, direction)

        # STT frames
        if self._show_stt:
            if isinstance(frame, InterimTranscriptionFrame):
                self._display.stt_interim(frame.text)
            elif isinstance(frame, TranscriptionFrame):
                # Get reason from result if available
                reason = ""
                if hasattr(frame, 'result') and frame.result:
                    reason = frame.result.get('reason', '')
                self._display.stt_final(frame.text, reason)

        # LLM frames
        if self._show_llm:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._in_llm_response = True
                self._display.llm_start()
            elif isinstance(frame, LLMTextFrame):
                if self._in_llm_response:
                    self._display.llm_chunk(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                if self._in_llm_response:
                    self._display.llm_end()
                    self._in_llm_response = False

        # TTS frames - show text being synthesized
        if self._show_tts:
            if isinstance(frame, TTSStartedFrame):
                pass  # Could add TTS start indicator
            elif isinstance(frame, TTSStoppedFrame):
                pass  # Could add TTS stop indicator

        # Pass frame through unchanged
        await self.push_frame(frame, direction)
