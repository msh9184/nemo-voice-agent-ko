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

import asyncio
import re
import time
import traceback
from typing import AsyncGenerator, List, Optional, Tuple

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pydantic import BaseModel

from nemo.agents.voice_agent.pipecat.services.nemo.streaming_asr import NemoStreamingASRService
from nemo.agents.voice_agent.pipecat.services.nemo.streaming_diar import NeMoStreamingDiarService

try:
    # disable nemo logging
    from nemo.utils import logging

    level = logging.getEffectiveLevel()
    logging.setLevel(logging.CRITICAL)


except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use NVIDIA NeMo STT, you need to `pip install "nemo_toolkit[all]"`.')
    raise Exception(f"Missing module: {e}")


class NeMoSTTInputParams(BaseModel):
    language: Optional[Language] = Language.EN_US
    att_context_size: Optional[List] = [70, 1]
    frame_len_in_secs: Optional[float] = 0.08  # 80ms for FastConformer model
    config_path: Optional[str] = None  # path to the Niva ASR config file
    raw_audio_frame_len_in_secs: Optional[float] = 0.016  # 16ms for websocket transport
    buffer_size: Optional[int] = 5  # number of audio frames to buffer, 1 frame is 16ms


class NemoSTTService(STTService):
    def __init__(
        self,
        *,
        model: Optional[str] = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi",
        device: Optional[str] = "cuda:0",
        sample_rate: Optional[int] = 16000,
        params: Optional[NeMoSTTInputParams] = None,
        has_turn_taking: bool = False,
        backend: Optional[str] = "legacy",
        decoder_type: Optional[str] = "rnnt",
        # Token filtering parameters
        filter_display_tokens: Optional[List[str]] = None,
        preserve_raw_for_llm: bool = True,
        extract_language_info: bool = True,
        # STT-only mode stability parameters
        stt_only_mode: bool = False,
        min_words_for_soft_reset: int = 3,
        max_words_for_soft_reset: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._queue = asyncio.Queue()
        self._sample_rate = sample_rate
        params.buffer_size = params.frame_len_in_secs // params.raw_audio_frame_len_in_secs
        self._params = params
        self._model_name = model
        self._has_turn_taking = has_turn_taking
        self._backend = backend
        self._decoder_type = decoder_type
        if not params:
            raise ValueError("params is required")

        self._device = device

        # Token filtering configuration
        self._filter_tokens = filter_display_tokens or []
        self._preserve_raw_for_llm = preserve_raw_for_llm
        self._extract_language_info = extract_language_info

        # STT-only mode stability configuration
        self._stt_only_mode = stt_only_mode
        self._min_words_for_soft_reset = min_words_for_soft_reset
        self._max_words_for_soft_reset = max_words_for_soft_reset
        # Track last boundary commit to prevent duplicate TranscriptionFrame for same text
        self._last_boundary_text = ""
        # Track committed text prefix for stripping from subsequent transcriptions
        # After reset_tokens_only(), encoder cache is preserved, so decoder may include
        # previously committed text as a prefix. We strip this to avoid overlap.
        self._committed_text_prefix = ""
        self._committed_prefix_timestamp = 0.0
        self._prefix_expiry_secs = 10.0  # Clear prefix if not seen for 10 seconds
        # Track max_words reset state to prevent cascading triggers
        # When max_words triggers reset, the stateful decoder continues accumulating text
        # (20→21→22→...). We skip further max_words triggers until word count drops
        # (new sentence) or significant increase (need another reset).
        self._last_max_words_reset_word_count = 0
        self._max_words_reset_interval = 10  # Only reset again after 10 more words
        if stt_only_mode:
            logger.info(f"STT-only mode enabled with min_words={min_words_for_soft_reset}, max_words={max_words_for_soft_reset}")

        # Pre-compile regex pattern for efficiency
        if self._filter_tokens:
            # Escape special regex characters in tokens
            escaped_tokens = [re.escape(t) for t in self._filter_tokens]
            self._filter_pattern = re.compile('|'.join(escaped_tokens))
            logger.info(f"STT token filter initialized with {len(self._filter_tokens)} tokens")
        else:
            self._filter_pattern = None

        self._load_model()

        self.audio_buffer = []

    def _load_model(self):
        if self._backend == "legacy":
            self._model = NemoStreamingASRService(
                self._model_name,
                device=self._device,
                decoder_type=self._decoder_type,
                att_context_size=self._params.att_context_size,
                chunk_size_in_secs=self._params.frame_len_in_secs,
            )
        else:
            raise ValueError(f"Invalid ASR backend: {self._backend}")

    def _filter_special_tokens(self, text: str) -> Tuple[str, str, Optional[str]]:
        """Filter special tokens from transcription text.

        Args:
            text: Raw transcription text potentially containing special tokens

        Returns:
            Tuple of (display_text, raw_text, detected_language)
            - display_text: Filtered text for UI display
            - raw_text: Original text with tokens (for LLM if needed)
            - detected_language: Extracted language code (e.g., "ko", "en") or None
        """
        if not text:
            return text, text, None

        raw_text = text
        detected_language = None

        # Extract language info before filtering
        if self._extract_language_info:
            # Look for language tokens like <lan_ko>, <lan_en>, etc.
            lang_match = re.search(r'<lan_(\w+)>', text)
            if lang_match:
                detected_language = lang_match.group(1)
                if detected_language != 'unk':  # Ignore unknown language
                    logger.debug(f"Detected language: {detected_language}")

        # Apply token filtering
        if self._filter_pattern:
            display_text = self._filter_pattern.sub('', text)
            # Clean up extra whitespace
            display_text = re.sub(r'\s+', ' ', display_text).strip()
        else:
            display_text = text

        return display_text, raw_text, detected_language

    def can_generate_metrics(self) -> bool:
        """Indicates whether this service can generate metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Handle service start.

        Args:
            frame: StartFrame containing initial configuration
        """
        await super().start(frame)

        # Initialize the model if not already done
        if not hasattr(self, "_model"):
            self._load_model()

    async def stop(self, frame: EndFrame):
        """Handle service stop.

        Args:
            frame: EndFrame that triggered this method
        """
        await super().stop(frame)
        # Clear any internal state if needed
        await self._queue.put(None)  # Signal to stop processing

    async def cancel(self, frame: CancelFrame):
        """Handle service cancellation.

        Args:
            frame: CancelFrame that triggered this method
        """
        await super().cancel(frame)
        # Clear any internal state
        await self._queue.put(None)  # Signal to stop processing
        self._queue = asyncio.Queue()  # Reset the queue

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data and generate transcription frames.

        Args:
            audio: Raw audio bytes to transcribe

        Yields:
            Frame: Transcription frames containing the results
        """
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

        # Debug: Log audio frame reception
        logger.debug(f"run_stt received audio: {len(audio)} bytes, buffer size: {len(self.audio_buffer)}/{self._params.buffer_size}")

        try:
            is_final = False
            transcription = None
            self.audio_buffer.append(audio)
            if len(self.audio_buffer) >= self._params.buffer_size:
                audio = b"".join(self.audio_buffer)
                self.audio_buffer = []

                logger.debug(f"Calling transcribe with {len(audio)} bytes of audio")
                asr_result = self._model.transcribe(audio)
                transcription = asr_result.text
                is_final = asr_result.is_final
                logger.debug(f"transcribe() returned: text='{transcription}', is_final={is_final}")
                eou_latency = asr_result.eou_latency
                eob_latency = asr_result.eob_latency
                eou_prob = asr_result.eou_prob
                eob_prob = asr_result.eob_prob
                if eou_latency is not None:
                    logger.debug(
                        f"EOU latency: {eou_latency: .4f} seconds. EOU probability: {eou_prob: .2f}."
                        f"Processing time: {asr_result.processing_time: .4f} seconds."
                    )
                if eob_latency is not None:
                    logger.debug(
                        f"EOB latency: {eob_latency: .4f} seconds. EOB probability: {eob_prob: .2f}."
                        f"Processing time: {asr_result.processing_time: .4f} seconds."
                    )
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()

            if transcription:
                # Apply token filtering for display vs LLM
                display_text, raw_text, detected_lang = self._filter_special_tokens(transcription)

                # STT-only mode: Strip committed prefix to avoid overlapping text
                # After reset_tokens_only(), encoder cache is preserved and decoder may
                # include previously committed text as a prefix
                if self._stt_only_mode:
                    if self._committed_text_prefix:
                        original_text = display_text
                        display_text = self._strip_committed_prefix(display_text)
                        if not display_text:
                            # Text was exactly the committed prefix, skip this frame
                            logger.debug(f"[Prefix] Skipped: matches committed prefix")
                            yield None
                            return
                        if display_text != original_text:
                            logger.debug(f"[Prefix] Applied: '{display_text[:40]}...'")

                logger.debug(f"STT Transcription (is_final={is_final}): `{display_text}`")
                if raw_text != display_text:
                    logger.debug(f"STT Raw (with tokens): `{raw_text}`")

                # Get the language from params or default to EN_US
                language = self._params.language if self._params else Language.EN_US

                # STT-only mode: Check for sentence boundary (punctuation + word count based)
                # This determines if we should mark transcription as final for client line break
                boundary_reason = ""
                if self._stt_only_mode and not is_final:
                    word_count = len(display_text.split())
                    potential_boundary = self._should_soft_reset(display_text, word_count)
                    if potential_boundary:
                        # Prevent duplicate boundary detection for the same/similar text
                        # This happens when reset_tokens_only() is called but encoder cache
                        # continues and outputs similar text with same punctuation
                        is_duplicate = self._is_duplicate_boundary(display_text)
                        if not is_duplicate:
                            boundary_reason = potential_boundary
                            # Store normalized version for consistent comparison later
                            self._last_boundary_text = self._normalize_text_for_comparison(display_text)
                            logger.debug(f"STT-only mode: Boundary detected ({boundary_reason}) at {word_count} words")
                        else:
                            logger.debug(f"STT-only mode: Skipping duplicate boundary")

                # Build result dict with filtering info
                result = {"text": display_text}
                if self._preserve_raw_for_llm and raw_text != display_text:
                    result["raw_text"] = raw_text
                if detected_lang:
                    result["detected_language"] = detected_lang

                # Add reason for finalization
                # IMPORTANT: max_words should NOT trigger FINAL (TranscriptionFrame)
                # max_words is a safety mechanism to prevent OOM by resetting tokens,
                # but it should NOT cause a line break in the client.
                # Only punctuation boundary ("boundary") and EOU should trigger FINAL.
                # This prevents the cascading FINAL issue where stateful decoder keeps
                # adding words (20→21→22→...) and each triggers max_words again.
                effective_is_final = is_final or (boundary_reason == "boundary")

                if is_final:
                    result["reason"] = "EOU"  # End-of-utterance token detected
                elif boundary_reason == "boundary":
                    result["reason"] = "boundary"  # Punctuation-based sentence boundary
                elif boundary_reason == "max_words":
                    result["reason"] = "max_words"  # Token reset only, not final

                # Create and push the transcription frame
                if self._has_turn_taking:
                    # if turn taking is enabled, we push interim transcription frames
                    # and let the turn taking service handle the final transcription
                    frame_type = InterimTranscriptionFrame
                elif self._stt_only_mode:
                    # STT-only mode: ALWAYS send InterimTranscriptionFrame
                    # The Aggregator is the authority on when to finalize (create TranscriptionFrame).
                    # This prevents duplicate TranscriptionFrames:
                    #   - If STT sends TranscriptionFrame, observer sees it
                    #   - Aggregator also creates TranscriptionFrame, observer sees it again
                    #   - Result: 2x duplicates at client
                    # By always sending Interim, only Aggregator's TranscriptionFrame reaches observer.
                    frame_type = InterimTranscriptionFrame
                else:
                    # Normal mode: Use effective_is_final for frame type
                    frame_type = InterimTranscriptionFrame if not effective_is_final else TranscriptionFrame

                logger.debug(f"Pushing {frame_type.__name__} to pipeline (effective_is_final={effective_is_final})")
                await self.push_frame(
                    frame_type(
                        display_text,  # Use filtered text for display
                        "",  # No speaker ID in this implementation
                        time_now_iso8601(),
                        language,
                        result=result,  # Contains raw_text, detected_language, and reason
                    )
                )

                # Handle the transcription (use display_text for downstream processing)
                await self._handle_transcription(
                    transcript=display_text,
                    is_final=effective_is_final,
                    language=language,
                )

                # STT-only mode: Perform soft reset if boundary was detected
                # This clears accumulated tokens but preserves encoder cache
                if boundary_reason:
                    # Directly call reset - condition already verified above
                    if hasattr(self._model, 'reset_tokens_only'):
                        logger.debug(
                            f"Performing soft reset ({boundary_reason}): "
                            f"text='{display_text[-30:] if len(display_text) > 30 else display_text}'"
                        )
                        self._model.reset_tokens_only()

                        # Store committed text as prefix to strip from subsequent outputs
                        # This prevents overlapping text when encoder cache continues
                        if boundary_reason == "boundary":
                            # Normalize prefix for consistent comparison later
                            self._committed_text_prefix = self._normalize_text_for_comparison(display_text)
                            self._committed_prefix_timestamp = time.time()
                            word_count = len(self._committed_text_prefix.split())
                            logger.debug(f"[Prefix] Set: {word_count} words")
            else:
                logger.debug(f"No transcription text returned (empty string)")

            yield None

        except Exception as e:
            logger.error(f"Error in NeMo STT processing: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            await self.push_frame(
                ErrorFrame(
                    str(e),
                    time_now_iso8601(),
                )
            )
            yield None

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Optional[str] = None):
        """Handle a transcription result.

        Args:
            transcript: The transcribed text
            is_final: Whether this is a final transcription
            language: The language of the transcription
        """
        pass  # Base implementation - can be extended for specific handling needs

    async def set_language(self, language: Language):
        """Update the service's recognition language.

        Args:
            language: New language for recognition
        """
        if self._params:
            self._params.language = language
        else:
            self._params = NeMoSTTInputParams(language=language)

        logger.info(f"Switching STT language to: {language}")

    async def set_model(self, model: str):
        """Update the service's model.

        Args:
            model: New model name/path to use
        """
        await super().set_model(model)
        self._model_name = model
        self._load_model()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # Reset the state of the model when end of utterance is detected by VAD
            # This is critical for models without EOU token detection (e.g., fastconformer)
            # Use reset_full() for hard reset (clears encoder cache)
            if hasattr(self._model, 'reset_full'):
                logger.debug("Resetting ASR state (full reset) due to VADUserStoppedSpeakingFrame")
                self._model.reset_full()
            elif hasattr(self._model, 'reset_state'):
                # Backward compatibility for older models
                logger.debug("Resetting ASR state (legacy) due to VADUserStoppedSpeakingFrame")
                self._model.reset_state()
            # Also clear our internal audio buffer and boundary tracking state
            self.audio_buffer = []
            # Reset boundary tracking so new sentences can trigger boundary detection
            self._last_boundary_text = ""
            # Clear committed prefix since this is a hard reset (new utterance)
            self._committed_text_prefix = ""
            # Reset max_words tracking for new utterance
            self._last_max_words_reset_word_count = 0
        await super().process_frame(frame, direction)

    def _is_duplicate_boundary(self, text: str) -> bool:
        """Check if this boundary text is a duplicate of the last committed boundary.

        After soft reset, the encoder cache continues and may output similar text
        with the same punctuation. This check prevents triggering boundary multiple
        times for what is essentially the same sentence.

        IMPORTANT: Uses WORD-LEVEL matching after normalization for robustness.
        This handles cases where SentencePiece markers or whitespace variations
        would cause character-level comparison to fail.

        Duplicate conditions (any match returns True):
        1. Exact word match with last boundary text
        2. New text starts with last boundary words (encoder continued with small addition)
        3. Last boundary starts with new text words (re-detection of same sentence)

        Args:
            text: Current transcription text to check

        Returns:
            True if this is a duplicate boundary that should be skipped
        """
        if not self._last_boundary_text:
            return False

        # Normalize both texts for consistent word-level comparison
        normalized_text = self._normalize_text_for_comparison(text)
        normalized_last = self._normalize_text_for_comparison(self._last_boundary_text)

        # Exact match (after normalization)
        if normalized_text == normalized_last:
            logger.debug(f"[Boundary] Duplicate: exact match")
            return True

        # Word-level prefix matching for robustness
        text_words = normalized_text.split()
        last_words = normalized_last.split()

        if not text_words or not last_words:
            return False

        # Find common prefix word count
        common_count = self._find_longest_common_prefix_words(normalized_text, normalized_last)

        # New text starts with last boundary (encoder continued)
        if len(text_words) > len(last_words):
            if common_count >= len(last_words) * 0.8:  # 80% word match
                new_words = text_words[common_count:]
                if len(new_words) < 5:  # Only small additions are duplicates
                    logger.debug(f"[Boundary] Duplicate: {common_count}/{len(last_words)} words overlap")
                    return True

        # Last boundary starts with new text (shorter re-detection)
        if len(last_words) > len(text_words):
            if common_count >= len(text_words) * 0.8:  # 80% word match
                logger.debug(f"[Boundary] Duplicate: shorter re-detection")
                return True

        return False

    def _normalize_text_for_comparison(self, text: str) -> str:
        """Normalize text for word-level comparison.

        Handles:
        - SentencePiece word boundary marker (▁ U+2581) → space
        - Unicode replacement characters (�)
        - Multiple consecutive whitespace → single space
        - Leading/trailing whitespace

        Args:
            text: Raw text from ASR

        Returns:
            Normalized text with consistent word boundaries
        """
        # Replace SentencePiece word boundary marker with space
        text = text.replace("\u2581", " ")  # ▁ → space
        # Remove Unicode replacement characters
        text = text.replace("\uFFFD", "")  # � (replacement character)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def _find_longest_common_prefix_words(self, text1: str, text2: str) -> int:
        """Find the number of common words at the start of two texts.

        Uses word-level comparison for robustness against slight character variations
        in ASR output (e.g., different word segmentation, whitespace).

        Args:
            text1: First text to compare (should be pre-normalized)
            text2: Second text to compare (should be pre-normalized)

        Returns:
            Number of common words at the start
        """
        words1 = text1.split()
        words2 = text2.split()

        common_count = 0
        for w1, w2 in zip(words1, words2):
            if w1 == w2:
                common_count += 1
            else:
                break

        return common_count

    def _strip_committed_prefix(self, text: str) -> str:
        """Strip previously committed text prefix from transcription.

        After reset_tokens_only(), the encoder cache is preserved. The stateful RNNT
        decoder continues from this cache and may include previously committed text
        as a prefix in new transcriptions. This method strips that prefix.

        Example:
            Committed: "이제 다도 체험을 하는데도 갔었는데요."
            New output: "이제 다도 체험을 하는데도 갔었는데요. 어/ 제가..."
            After strip: "어/ 제가..."

        IMPORTANT: Uses WORD-LEVEL matching for robustness against ASR variations.
        The ASR model may output slightly different word segmentation between frames.
        Character-level startswith() can fail due to these variations.

        Matching strategy:
        1. Find longest common prefix (word-level)
        2. If common words >= 50% of prefix words → STRIP common portion
        3. If text is shorter and mostly matches prefix → PARTIAL, keep waiting
        4. If no significant overlap → New utterance, clear prefix

        Args:
            text: Current transcription text from decoder

        Returns:
            Text with committed prefix stripped, or original if no prefix match
        """
        if not self._committed_text_prefix:
            return text

        current_time = time.time()

        # Check if prefix has expired (too long since last seen)
        elapsed = current_time - self._committed_prefix_timestamp
        if elapsed > self._prefix_expiry_secs:
            logger.debug(f"[Prefix] Expired after {elapsed:.0f}s")
            self._committed_text_prefix = ""
            return text

        # Normalize text for comparison (handles SentencePiece ▁ markers)
        normalized_text = self._normalize_text_for_comparison(text)
        normalized_prefix = self._normalize_text_for_comparison(self._committed_text_prefix)

        # Split into words for word-level matching
        text_words = normalized_text.split()
        prefix_words = normalized_prefix.split()

        if not text_words:
            return text

        # Find longest common prefix (word-level)
        common_word_count = self._find_longest_common_prefix_words(normalized_text, normalized_prefix)

        # Calculate overlap ratios
        prefix_coverage = common_word_count / len(prefix_words) if prefix_words else 0
        text_coverage = common_word_count / len(text_words) if text_words else 0

        logger.debug(
            f"[Prefix] Check: text={len(text_words)}w, prefix={len(prefix_words)}w, "
            f"common={common_word_count}, coverage={prefix_coverage:.0%}"
        )

        # Case 1: Text has enough common words AND has new content after
        # Strip if we've matched most of the prefix (>= 80% coverage)
        if prefix_coverage >= 0.8 and len(text_words) > common_word_count:
            remaining_words = text_words[common_word_count:]
            stripped = ' '.join(remaining_words)
            if stripped:
                logger.debug(f"[Prefix] Stripped {common_word_count} words → '{stripped[:40]}...'")
                self._committed_prefix_timestamp = current_time
                return stripped

        # Case 2: Text matches prefix exactly or is subset (PARTIAL - keep waiting)
        if text_coverage >= 0.8 and len(text_words) <= len(prefix_words):
            logger.debug(f"[Prefix] Partial match: {common_word_count}/{len(text_words)} words")
            self._committed_prefix_timestamp = current_time
            return ""  # Skip duplicate/partial content

        # Case 3: Some overlap but text is mostly beyond prefix
        if common_word_count > 0 and prefix_coverage >= 0.5:
            remaining_words = text_words[common_word_count:]
            stripped = ' '.join(remaining_words)
            if stripped:
                logger.debug(f"[Prefix] Partial strip {common_word_count} words → '{stripped[:40]}...'")
                self._committed_prefix_timestamp = current_time
                return stripped

        # Case 4: No significant overlap → New utterance
        if common_word_count == 0:
            logger.debug(f"[Prefix] No overlap - clearing prefix (new utterance)")
            self._committed_text_prefix = ""
            return normalized_text

        # Case 5: Low overlap - ambiguous, keep prefix but return text as-is
        logger.debug(f"[Prefix] Low overlap ({common_word_count} words), keeping prefix")
        self._committed_prefix_timestamp = current_time
        return normalized_text

    def _should_soft_reset(self, text: str, word_count: int) -> str:
        """Check if soft reset conditions are met and return the reason.

        For STT-only mode, we perform soft reset (tokens only, preserve encoder cache)
        when a sentence boundary is detected. This keeps STT's token state in sync
        with the downstream Aggregator's buffer state.

        Boundary conditions (in priority order):
        1. punctuation + min_words (normal sentence boundary) - triggers FINAL
        2. max_words reached (safety net for very long utterances) - triggers reset only

        IMPORTANT: max_words boundary is rate-limited to prevent cascading triggers.
        After reset_tokens_only(), the stateful decoder continues from encoder cache,
        causing word count to grow (20→21→22→...). We only trigger max_words reset
        again after word count drops (new sentence) or increases significantly.

        Args:
            text: Current transcription text
            word_count: Number of words in the text

        Returns:
            Reason string if soft reset should be performed, empty string otherwise.
            Possible values: "max_words", "boundary", ""
        """
        text = text.strip()
        if not text:
            return ""

        # Check punctuation boundary FIRST (has priority, triggers FINAL)
        if word_count >= self._min_words_for_soft_reset:
            # Check for sentence-ending punctuation (including Korean punctuation)
            sentence_end_chars = '.?!。？！'
            if text[-1] in sentence_end_chars:
                # Avoid false positives for decimal numbers (e.g., "3.14")
                if text[-1] == '.' and len(text) > 1 and text[-2].isdigit():
                    pass  # Not a sentence boundary
                else:
                    # Reset max_words tracking on punctuation boundary
                    self._last_max_words_reset_word_count = 0
                    return "boundary"

        # Check max words boundary (safety net, reset only, no FINAL)
        if word_count >= self._max_words_for_soft_reset:
            # Check if word count dropped (new sentence started after VAD)
            if word_count < self._last_max_words_reset_word_count:
                # Word count dropped, this is a new sentence - reset tracking
                self._last_max_words_reset_word_count = 0

            # Rate-limit max_words triggers to prevent cascading resets
            # Only trigger again after word count increases by _max_words_reset_interval
            words_since_last_reset = word_count - self._last_max_words_reset_word_count
            if self._last_max_words_reset_word_count == 0 or words_since_last_reset >= self._max_words_reset_interval:
                logger.debug(f"Max words boundary: {word_count} words (last reset at {self._last_max_words_reset_word_count})")
                self._last_max_words_reset_word_count = word_count
                return "max_words"
            else:
                logger.debug(f"Skipping max_words: {word_count} words, only {words_since_last_reset} since last reset")

        return ""

    async def _perform_soft_reset_if_needed(self, text: str) -> bool:
        """Perform soft reset if STT-only mode and conditions are met.

        Soft reset clears accumulated tokens but preserves encoder cache
        for continuous streaming without recognition accuracy loss.

        Args:
            text: Current transcription text

        Returns:
            True if soft reset was performed (sentence boundary detected)
        """
        if not self._stt_only_mode:
            return False

        # Count words directly from current text (NOT cumulative)
        # The text parameter is already the accumulated transcription
        word_count = len(text.split())

        # Check if soft reset should be performed
        if self._should_soft_reset(text, word_count):
            if hasattr(self._model, 'reset_tokens_only'):
                logger.debug(
                    f"Performing soft reset: {word_count} words, "
                    f"text ends with '{text[-20:] if len(text) > 20 else text}'"
                )
                self._model.reset_tokens_only()
                return True  # Sentence boundary detected

        return False
