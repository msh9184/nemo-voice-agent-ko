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
import inspect
import io
from collections.abc import AsyncGenerator
from typing import Iterator, List, Optional

import numpy as np
import torch
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService

from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# Optional aiohttp for async streaming TTS
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available. Fish Speech API streaming will use fallback mode.")


class BaseNemoTTSService(TTSService):
    """Text-to-Speech service using Nemo TTS models.

    This service works with any TTS model that exposes a generate(text) method
    that returns audio data. The TTS generation runs in a dedicated background thread to
    avoid blocking the main asyncio event loop, following the same pattern as NemoDiarService.

    Args:
        model: TTS model instance with a generate(text) method
        sample_rate: Audio sample rate in Hz (defaults to 22050)
        **kwargs: Additional arguments passed to TTSService
    """

    def __init__(
        self,
        *,
        model,
        device: str = "cuda",
        sample_rate: int = 22050,
        think_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_name = model
        self._device = device
        self._model = self._setup_model()
        self._think_tokens = think_tokens
        if think_tokens is not None:
            assert (
                isinstance(think_tokens, list) and len(think_tokens) == 2
            ), f"think_tokens must be a list of two strings: {think_tokens}"

        # Background processing infrastructure - no response handler needed
        self._tts_queue = asyncio.Queue()
        self._processing_task = None
        self._processing_running = False

        # Track pending requests with their response queues
        self._pending_requests = {}
        self._have_seen_think_tokens = False

    def _setup_model(self):
        raise NotImplementedError("Subclass must implement _setup_model")

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        raise NotImplementedError("Subclass must implement _generate_audio")

    def can_generate_metrics(self) -> bool:
        """If the TTS service can generate metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Handle service start."""
        await super().start(frame)

        # Initialize the model if not already done
        if not hasattr(self, "_model") or self._model is None:
            self._model = self._setup_model()

        # Only start background processing task - no response handler needed
        if not self._processing_task:
            self._processing_task = self.create_task(self._processing_task_handler())

    async def stop(self, frame: EndFrame):
        """Handle service stop."""
        await super().stop(frame)
        await self._stop_tasks()

    async def cancel(self, frame: CancelFrame):
        """Handle service cancellation."""
        await super().cancel(frame)
        await self._stop_tasks()

    async def _stop_tasks(self):
        """Stop background processing tasks."""
        self._processing_running = False
        await self._tts_queue.put(None)  # Signal to stop processing

        if self._processing_task:
            await self.cancel_task(self._processing_task)
            self._processing_task = None

    def _tts_processor(self):
        """Background processor that handles TTS generation calls."""
        try:
            while self._processing_running:
                try:
                    future = asyncio.run_coroutine_threadsafe(self._tts_queue.get(), self.get_event_loop())
                    request = future.result()

                    if request is None:  # Stop signal
                        logger.debug("Received stop signal in TTS background processor")
                        break

                    text, request_id = request
                    logger.debug(f"Processing TTS request for text: [{text}]")

                    # Get the response queue for this request
                    response_queue = None
                    future = asyncio.run_coroutine_threadsafe(
                        self._get_response_queue(request_id), self.get_event_loop()
                    )
                    response_queue = future.result()

                    if response_queue is None:
                        logger.warning(f"No response queue found for request {request_id}")
                        continue

                    # Process TTS generation
                    try:
                        audio_result = self._generate_audio(text)

                        # Send result directly to the waiting request
                        asyncio.run_coroutine_threadsafe(
                            response_queue.put(('success', audio_result)), self.get_event_loop()
                        )
                    except Exception as e:
                        logger.error(f"Error in TTS generation: {e}")
                        # Send error directly to the waiting request
                        asyncio.run_coroutine_threadsafe(response_queue.put(('error', e)), self.get_event_loop())

                except Exception as e:
                    logger.error(f"Error in background TTS processor: {e}")

        except Exception as e:
            logger.error(f"Background TTS processor fatal error: {e}")
        finally:
            logger.debug("Background TTS processor stopped")

    async def _get_response_queue(self, request_id: str):
        """Get the response queue for a specific request."""
        return self._pending_requests.get(request_id)

    async def _processing_task_handler(self):
        """Handler for background processing task."""
        try:
            self._processing_running = True
            logger.debug("Starting background TTS processing task")
            await asyncio.to_thread(self._tts_processor)
        except asyncio.CancelledError:
            logger.debug("Background TTS processing task cancelled")
            self._processing_running = False
            raise
        finally:
            self._processing_running = False

    def _handle_think_tokens(self, text: str) -> Optional[str]:
        """
        Handle the thinking tokens for TTS.
        If the thinking tokens are not provided, return the text as it is.
        Otherwise:
            If both thinking tokens appear in the text, return the text after the end of thinking tokens.
            If the LLM is thinking, return None.
            If the LLM is done thinking, return the text after the end of thinking tokens.
            If the LLM starts thinking, return the text before the start of thinking tokens.
            If the LLM is not thinking, return the text as is.
        """
        if not self._think_tokens:
            return text
        elif self._think_tokens[0] in text and self._think_tokens[1] in text:
            # LLM finishes thinking in one chunk or outputs dummy thinking tokens
            logger.debug(f"LLM finishes thinking: {text}")
            idx = text.index(self._think_tokens[1])
            # only return the text after the end of thinking tokens
            text = text[idx + len(self._think_tokens[1]) :]
            self._have_seen_think_tokens = False
            logger.debug(f"Returning text after thinking: {text}")
            return text
        elif self._have_seen_think_tokens:
            # LLM is thinking
            if self._think_tokens[1] not in text:
                logger.debug(f"LLM is still thinking: {text}")
                # LLM is still thinking
                return None
            else:
                # LLM is done thinking
                logger.debug(f"LLM is done thinking: {text}")
                idx = text.index(self._think_tokens[1])
                # only return the text after the end of thinking tokens
                text = text[idx + len(self._think_tokens[1]) :]
                self._have_seen_think_tokens = False
                logger.debug(f"Returning text after thinking: {text}")
                return text
        elif self._think_tokens[0] in text:
            # LLM now starts thinking
            logger.debug(f"LLM starts thinking: {text}")
            self._have_seen_think_tokens = True
            # return text before the start of thinking tokens
            idx = text.index(self._think_tokens[0])
            text = text[:idx]
            logger.debug(f"Returning text before thinking: {text}")
            return text
        else:
            # LLM is not thinking
            return text

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the Nemo TTS model."""
        text = self._handle_think_tokens(text)

        if not text:
            yield None
            return

        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            # Generate unique request ID
            import uuid

            request_id = str(uuid.uuid4())

            # Create response queue for this specific request
            request_queue = asyncio.Queue()
            self._pending_requests[request_id] = request_queue

            try:
                # Queue the TTS request for background processing
                await self._tts_queue.put((text, request_id))

                # Wait for the result directly from our request queue
                result = await request_queue.get()
                status, data = result

                if status == 'error':
                    logger.error(f"{self} TTS generation error: {data}")
                    yield ErrorFrame(error=f"TTS generation error: {str(data)}")
                    return

                audio_result = data
                if audio_result is None:
                    logger.error(f"{self} TTS model returned None for text: [{text}]")
                    yield ErrorFrame(error="TTS generation failed - no audio returned")
                    return

                await self.start_tts_usage_metrics(text)

                # Process the audio result (same as before)
                if (
                    inspect.isgenerator(audio_result)
                    or hasattr(audio_result, '__iter__')
                    and hasattr(audio_result, '__next__')
                ):
                    # Handle generator case
                    first_chunk = True
                    for audio_chunk in audio_result:
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False

                        if audio_chunk is None:
                            break

                        audio_bytes = self._convert_to_bytes(audio_chunk)
                        chunk_size = self.chunk_size
                        for i in range(0, len(audio_bytes), chunk_size):
                            audio_chunk_bytes = audio_bytes[i : i + chunk_size]
                            if not audio_chunk_bytes:
                                break

                            frame = TTSAudioRawFrame(
                                audio=audio_chunk_bytes, sample_rate=self.sample_rate, num_channels=1
                            )
                            yield frame
                else:
                    # Handle single result case
                    await self.stop_ttfb_metrics()
                    audio_bytes = self._convert_to_bytes(audio_result)

                    chunk_size = self.chunk_size
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i : i + chunk_size]
                        if not chunk:
                            break

                        frame = TTSAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)
                        yield frame

                yield TTSStoppedFrame()

            finally:
                # Clean up the pending request
                if request_id in self._pending_requests:
                    del self._pending_requests[request_id]

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            error_message = f"TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)

    def _convert_to_bytes(self, audio_data) -> bytes:
        """Convert various audio data formats to bytes."""
        if isinstance(audio_data, (bytes, bytearray)):
            return bytes(audio_data)

        # Handle numpy arrays
        try:
            import numpy as np

            if isinstance(audio_data, np.ndarray):
                # Ensure it's in the right format (16-bit PCM)
                if audio_data.dtype in [np.float32, np.float64]:
                    # Convert float [-1, 1] to int16 [-32768, 32767]
                    audio_data = np.clip(audio_data, -1.0, 1.0)  # Ensure values are in range
                    audio_data = (audio_data * 32767).astype(np.int16)
                elif audio_data.dtype != np.int16:
                    # Convert other integer types to int16
                    audio_data = audio_data.astype(np.int16)
                return audio_data.tobytes()
            elif hasattr(audio_data, 'tobytes'):
                return audio_data.tobytes()
            else:
                return bytes(audio_data)
        except ImportError:
            # Fallback if numpy is not available
            if hasattr(audio_data, 'tobytes'):
                return audio_data.tobytes()
            else:
                return bytes(audio_data)


class NeMoFastPitchHiFiGANTTSService(BaseNemoTTSService):
    """Text-to-Speech service using NeMo FastPitch-Hifigan model.

    More info: https://huggingface.co/nvidia/tts_en_fastpitch

    Args:
        fastpitch_model: FastPitch model name
        hifigan_model: Hifigan model name
        device: Device to run on (default: 'cuda')
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        fastpitch_model: str = "nvidia/tts_en_fastpitch",
        hifigan_model: str = "nvidia/tts_hifigan",
        device: str = "cuda",
        **kwargs,
    ):
        model_name = f"{fastpitch_model}+{hifigan_model}"
        self._fastpitch_model_name = fastpitch_model
        self._hifigan_model_name = hifigan_model
        super().__init__(model=model_name, device=device, **kwargs)

    def _setup_model(self):
        print("Loading model...")
        self._fastpitch_model = self._setup_fastpitch_model(self._fastpitch_model_name)
        self._hifigan_model = self._setup_hifigan_model(self._hifigan_model_name)
        return self._fastpitch_model, self._hifigan_model

    def _setup_fastpitch_model(self, model_name: str):
        if model_name.endswith(".nemo"):
            fastpitch_model = FastPitchModel.restore_from(model_name, map_location=torch.device(self._device))
        else:
            fastpitch_model = FastPitchModel.from_pretrained(model_name, map_location=torch.device(self._device))
        fastpitch_model.eval()
        return fastpitch_model

    def _setup_hifigan_model(self, model_name: str):
        if model_name.endswith(".nemo"):
            hifigan_model = HifiGanModel.restore_from(model_name, map_location=torch.device(self._device))
        else:
            hifigan_model = HifiGanModel.from_pretrained(model_name, map_location=torch.device(self._device))
        hifigan_model.eval()
        return hifigan_model

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        with torch.no_grad():
            parsed = self._fastpitch_model.parse(text)
            spectrogram = self._fastpitch_model.generate_spectrogram(tokens=parsed)
            audio = self._hifigan_model.convert_spectrogram_to_audio(spec=spectrogram)
            audio = audio.detach().view(-1).cpu().numpy()
            yield audio


class KokoroTTSService(BaseNemoTTSService):
    """Text-to-Speech service using Kokoro-82M model.

    Kokoro is an open-weight TTS model with 82 million parameters.
    More info: https://huggingface.co/hexgrad/Kokoro-82M

    Args:
        lang_code: Language code for the model (default: 'a' for American English)
        voice: Voice to use (default: 'af_heart')
        device: Device to run on (default: 'cuda')
        sample_rate: Audio sample rate in Hz (default: 24000 for Kokoro)
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        lang_code: str = "a",
        voice: str = "af_heart",
        device: str = "cuda",
        sample_rate: int = 24000,
        speed: float = 1.0,
        **kwargs,
    ):
        self._lang_code = lang_code
        self._voice = voice
        self._speed = speed
        model_name = f"kokoro-{lang_code}-{voice}"
        super().__init__(model=model_name, device=device, sample_rate=sample_rate, **kwargs)

    def _setup_model(self):
        """Initialize the Kokoro pipeline."""
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "kokoro package is required for KokoroTTSService. " "Install it with: pip install kokoro>=0.9.2"
            )

        logger.info(f"Loading Kokoro TTS model with lang_code={self._lang_code}, voice={self._voice}")
        pipeline = KPipeline(lang_code=self._lang_code)
        return pipeline

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Generate audio using the Kokoro pipeline.

        Args:
            text: Text to convert to speech

        Yields:
            Audio data as numpy arrays
        """
        try:
            # Generate audio using Kokoro pipeline
            generator = self._model(text, voice=self._voice, speed=self._speed)

            # The generator yields tuples of (gs, ps, audio)
            # We only need the audio component
            for i, (gs, ps, audio) in enumerate(generator):
                logger.debug(
                    f"Kokoro generated audio chunk {i}: gs={gs}, ps={ps},"
                    f"audio_shape={audio.shape if hasattr(audio, 'shape') else len(audio)}"
                )
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                # Kokoro returns audio as numpy array in float32 format [-1, 1]
                # The base class will handle conversion to int16 bytes
                yield audio

        except Exception as e:
            logger.error(f"Error generating audio with Kokoro: {e}")
            raise


class MeloTTSKoreanService(BaseNemoTTSService):
    """Text-to-Speech service using MeloTTS Korean model.

    MeloTTS is a high-quality multi-lingual text-to-speech library by MyShell.ai.
    This service uses the Korean language model for natural Korean speech synthesis.
    More info: https://huggingface.co/myshell-ai/MeloTTS-Korean

    Features:
        - Native Korean text support (no romanization needed)
        - CPU real-time inference capable
        - VITS-based architecture for high quality
        - MIT license (commercial use allowed)

    Args:
        device: Device to run on (default: 'cpu' - MeloTTS is CPU real-time capable)
        sample_rate: Audio sample rate in Hz (default: 44100 for MeloTTS)
        speed: Speaking rate multiplier (default: 1.0)
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        device: str = "cpu",
        sample_rate: int = 44100,
        speed: float = 1.0,
        **kwargs,
    ):
        self._speed = speed
        model_name = "melo-tts-korean"
        super().__init__(model=model_name, device=device, sample_rate=sample_rate, **kwargs)

    def _setup_model(self):
        """Initialize the MeloTTS Korean model."""
        try:
            from melo.api import TTS
        except ImportError:
            raise ImportError(
                "melo package is required for MeloTTSKoreanService. "
                "Install it with: pip install melotts"
            )

        logger.info(f"Loading MeloTTS Korean model on device={self._device}")
        model = TTS(language='KR', device=self._device)
        self._speaker_id = model.hps.data.spk2id['KR']
        logger.info(f"MeloTTS Korean model loaded successfully. Speaker ID: {self._speaker_id}")
        return model

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Generate audio using the MeloTTS Korean model.

        Args:
            text: Korean text to convert to speech

        Yields:
            Audio data as numpy arrays (chunked for streaming)
        """
        try:
            logger.debug(f"Generating Korean TTS for text: [{text}]")

            # MeloTTS generates complete audio, we need to handle it for streaming
            # Use the internal method to get raw audio without saving to file
            audio = self._model.tts_to_file(
                text,
                self._speaker_id,
                output_path=None,  # Don't save to file, return audio directly
                speed=self._speed,
            )

            # If audio is a torch tensor, convert to numpy
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()

            # MeloTTS returns audio as numpy array in float32 format
            if audio is not None and len(audio) > 0:
                logger.debug(f"MeloTTS generated audio shape: {audio.shape}, dtype: {audio.dtype}")

                # Chunk the audio for streaming (4096 samples per chunk at 44.1kHz ≈ 93ms)
                chunk_size = 4096
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) > 0:
                        yield chunk
            else:
                logger.warning(f"MeloTTS returned empty audio for text: [{text}]")

        except Exception as e:
            logger.error(f"Error generating audio with MeloTTS Korean: {e}")
            raise


class CosyVoice3KoreanService(BaseNemoTTSService):
    """Text-to-Speech service using CosyVoice3 (Fun-CosyVoice3-0.5B) for Korean.

    CosyVoice3 is an advanced multilingual speech synthesis model by Alibaba FunAudioLLM,
    offering 68.7% relative improvement in Korean compared to previous versions.
    Supports zero-shot voice cloning with just 3-10 seconds of reference audio.

    More info: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
    GitHub: https://github.com/FunAudioLLM/CosyVoice

    Features:
        - 9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
        - Zero-shot voice cloning (3-10s reference audio)
        - Cross-lingual synthesis with <|ko|> language tag
        - Streaming support with 150ms latency
        - Instruction-based control (emotion, speed, dialect)
        - 0.5B parameters, Apache 2.0 license

    Model files (~7.45GB total):
        - llm.pt (2.02GB), flow.pt (1.33GB), hift.pt (83.2MB)
        - speech_tokenizer_v3.onnx (969MB)

    CRITICAL for Korean TTS:
        1. text_frontend=False - MUST be disabled (only supports Chinese/English)
        2. prompt_text - MUST match the actual transcript of reference_audio
        3. Reference audio should be 16kHz, 3-10 seconds of clean Korean speech

    Args:
        model_path: Path to CosyVoice3 model directory
        device: Device to run on (default: 'cuda')
        sample_rate: Audio sample rate in Hz (default: 22050 for CosyVoice3)
        reference_audio_path: Optional path to reference audio for voice cloning (3-10s, 16kHz)
        reference_audio_text: The EXACT transcript of what the reference audio says (CRITICAL!)
        use_instruct: Use instruction mode for emotion/style control
        streaming: Enable streaming inference (default: True)
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        model_path: str = "pretrained_models/Fun-CosyVoice3-0.5B",
        device: str = "cuda",
        sample_rate: int = 22050,
        reference_audio_path: Optional[str] = None,
        reference_audio_text: Optional[str] = None,
        use_instruct: bool = False,
        streaming: bool = True,
        **kwargs,
    ):
        self._model_path = model_path
        self._reference_audio_path = reference_audio_path
        self._reference_audio_text = reference_audio_text or ""  # Transcript of reference audio
        self._reference_audio = None
        self._use_instruct = use_instruct
        self._streaming = streaming
        model_name = "cosyvoice3-korean"
        super().__init__(model=model_name, device=device, sample_rate=sample_rate, **kwargs)

    def _setup_model(self):
        """Initialize the CosyVoice3 model."""
        try:
            # Try new AutoModel API first (CosyVoice3)
            try:
                from cosyvoice.cli.model import AutoModel
                logger.info(f"Loading CosyVoice3 model from {self._model_path}")
                model = AutoModel(model_dir=self._model_path)
            except ImportError:
                # Fallback to older CosyVoice3 API
                from cosyvoice.cli.cosyvoice import CosyVoice3
                logger.info(f"Loading CosyVoice3 model from {self._model_path}")
                model = CosyVoice3(self._model_path)
        except ImportError as e:
            raise ImportError(
                "cosyvoice package is required for CosyVoice3KoreanService. "
                "Install from: https://github.com/FunAudioLLM/CosyVoice\n"
                f"Error: {e}"
            )

        # Validate reference audio path if provided
        # IMPORTANT: prompt_wav expects FILE PATH (string), not pre-loaded tensor!
        # CosyVoice internally calls load_wav(prompt_wav, 24000) to load and resample
        if self._reference_audio_path:
            import os
            if os.path.exists(self._reference_audio_path):
                # Store the path - CosyVoice will load it internally
                self._reference_audio = self._reference_audio_path
                logger.info(f"Reference audio path validated: {self._reference_audio_path}")
                if self._reference_audio_text:
                    logger.info(f"Reference audio transcript: '{self._reference_audio_text}'")
                else:
                    logger.warning(
                        "WARNING: reference_audio_text not provided! "
                        "For best Korean TTS quality, provide the exact transcript of the reference audio. "
                        "Without it, voice cloning quality may be degraded."
                    )
            else:
                logger.warning(f"Reference audio file not found: {self._reference_audio_path}")
                self._reference_audio = None

        logger.info(f"CosyVoice3 model loaded successfully. Sample rate: {self.sample_rate}")
        return model

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Generate audio using the CosyVoice3 model.

        For Korean, uses zero-shot inference with reference audio and its transcript.

        CRITICAL for Korean TTS (must follow ALL):
            1. Use inference_cross_lingual (NOT inference_zero_shot) for Korean!
            2. Add <|ko|> language tag to tts_text (REQUIRED for cross-lingual)
            3. text_frontend=False - Disable text frontend (only supports Chinese/English)
            4. prompt_wav - FILE PATH (string) to reference audio

            Note: inference_zero_shot is for SAME language as reference audio.
            inference_cross_lingual is for DIFFERENT languages (Korean, English, etc.)

        Reference: https://huggingface.co/FunAudioLLM/CosyVoice3-0.5B-2512

        Args:
            text: Korean text to convert to speech

        Yields:
            Audio data as numpy arrays
        """
        try:
            logger.debug(f"Generating Korean TTS with CosyVoice3 for text: [{text}]")

            if self._reference_audio is not None:
                # Cross-lingual mode - THE CORRECT MODE FOR KOREAN!
                # - inference_zero_shot: SAME language as reference, NO language tag
                # - inference_cross_lingual: DIFFERENT language, REQUIRES <|ko|> tag

                # Add Korean language tag
                text_with_tag = f"<|ko|>{text}"

                logger.debug(f"Using cross-lingual mode with text: {text_with_tag}")
                logger.debug(f"Using prompt_wav (file path): {self._reference_audio}")

                generator = self._model.inference_cross_lingual(
                    tts_text=text_with_tag,  # Korean text WITH <|ko|> tag!
                    prompt_wav=self._reference_audio,  # FILE PATH (string)
                    stream=self._streaming,
                    text_frontend=False  # REQUIRED for Korean!
                )
            elif self._use_instruct:
                # Instruction mode for style control
                # Note: instruct mode may have limited Korean support
                instruct_text = "한국어로 자연스럽게 말해주세요."
                generator = self._model.inference_instruct(
                    tts_text=text,
                    instruct_text=instruct_text,
                    stream=self._streaming,
                    text_frontend=False  # CRITICAL: Disable text frontend for Korean!
                )
            else:
                # Cross-lingual mode with Korean language tag
                # Note: This mode requires reference audio for best results
                text_with_tag = f"<|ko|>{text}"
                logger.debug(f"Using cross-lingual mode with text: {text_with_tag}")
                logger.warning(
                    "Cross-lingual mode without reference audio may produce suboptimal results. "
                    "For best Korean TTS, provide reference_audio_path and reference_audio_text."
                )

                # Try cross-lingual first, fallback to zero-shot with default voice
                try:
                    generator = self._model.inference_cross_lingual(
                        tts_text=text_with_tag,
                        prompt_wav=None,  # Correct parameter name!
                        stream=self._streaming,
                        text_frontend=False  # CRITICAL: Disable text frontend for Korean!
                    )
                except (AttributeError, TypeError):
                    # Model may not support cross_lingual without reference
                    # Use SFT mode if available
                    logger.debug("Falling back to SFT inference mode")
                    generator = self._model.inference_sft(
                        tts_text=text,
                        spk_id="default",  # Will use default speaker
                        stream=self._streaming,
                        text_frontend=False  # CRITICAL: Disable text frontend for Korean!
                    )

            # Process generator output
            for result in generator:
                if isinstance(result, dict):
                    audio = result.get('tts_speech', result.get('audio', None))
                else:
                    audio = result

                if audio is not None:
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()

                    # Flatten if needed
                    if audio.ndim > 1:
                        audio = audio.squeeze()

                    # Chunk for streaming
                    chunk_size = 4096
                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i:i + chunk_size]
                        if len(chunk) > 0:
                            yield chunk

        except Exception as e:
            logger.error(f"Error generating audio with CosyVoice3: {e}")
            import traceback
            traceback.print_exc()
            raise


class CosyVoiceKoreanService(BaseNemoTTSService):
    """Text-to-Speech service using CosyVoice (legacy version) for Korean.

    This is the legacy CosyVoice service. For better Korean performance,
    use CosyVoice3KoreanService instead.

    More info: https://github.com/FunAudioLLM/CosyVoice

    Args:
        model_path: Path to the CosyVoice model
        device: Device to run on (default: 'cuda')
        sample_rate: Audio sample rate in Hz (default: 22050)
        reference_audio: Optional path to reference audio for voice cloning
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        model_path: str = "pretrained_models/CosyVoice-300M",
        device: str = "cuda",
        sample_rate: int = 22050,
        reference_audio: Optional[str] = None,
        **kwargs,
    ):
        self._model_path = model_path
        self._reference_audio = reference_audio
        model_name = "cosyvoice-korean"
        super().__init__(model=model_name, device=device, sample_rate=sample_rate, **kwargs)

    def _setup_model(self):
        """Initialize the CosyVoice model."""
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
        except ImportError:
            raise ImportError(
                "cosyvoice package is required for CosyVoiceKoreanService. "
                "Install it from: https://github.com/FunAudioLLM/CosyVoice"
            )

        logger.info(f"Loading CosyVoice model from {self._model_path} on device={self._device}")
        model = CosyVoice(self._model_path)
        logger.info("CosyVoice model loaded successfully")
        return model

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Generate audio using the CosyVoice model.

        IMPORTANT: text_frontend=False is REQUIRED for Korean!
        The CosyVoice text frontend only supports Chinese and English normalization.

        Args:
            text: Korean text to convert to speech

        Yields:
            Audio data as numpy arrays
        """
        try:
            logger.debug(f"Generating Korean TTS with CosyVoice for text: [{text}]")

            if self._reference_audio:
                from cosyvoice.utils.file_utils import load_wav
                prompt_speech = load_wav(self._reference_audio, 16000)

                # Zero-shot voice cloning mode with Korean tag
                text_with_tag = f"<|ko|>{text}"
                generator = self._model.inference_zero_shot(
                    text_with_tag,
                    "",
                    prompt_speech,
                    stream=True,
                    text_frontend=False  # CRITICAL: Disable text frontend for Korean!
                )
            else:
                # Cross-lingual mode
                text_with_tag = f"<|ko|>{text}"
                generator = self._model.inference_cross_lingual(
                    text_with_tag,
                    None,
                    stream=True,
                    text_frontend=False  # CRITICAL: Disable text frontend for Korean!
                )

            for result in generator:
                if isinstance(result, dict):
                    audio = result.get('tts_speech', None)
                else:
                    audio = result

                if audio is not None:
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
                    yield audio

        except Exception as e:
            logger.error(f"Error generating audio with CosyVoice: {e}")
            raise


class FishSpeechKoreanService(BaseNemoTTSService):
    """Text-to-Speech service using OpenAudio S1-mini for Korean.

    OpenAudio S1-mini is a multilingual TTS model with strong Korean support:
    - 0.5B parameters (distilled from 4B S1 model)
    - 2M+ hours training data with RLHF
    - Zero-shot voice cloning with 10-30s reference audio
    - 13 languages: EN, ZH, JA, KO, DE, FR, ES, AR, RU, NL, IT, PL, PT
    - 45+ emotion markers support

    More info:
    - GitHub: https://github.com/fishaudio/fish-speech
    - HuggingFace: https://huggingface.co/fishaudio/openaudio-s1-mini
    - Docs: https://speech.fish.audio/

    Args:
        model_path: Path to OpenAudio S1-mini model directory
        device: Device to run on (default: 'cuda')
        sample_rate: Audio sample rate in Hz (default: 24000)
        reference_audio_path: Path to reference audio for voice cloning (10-30s)
        reference_audio_text: Transcript of the reference audio
        streaming: Enable streaming inference
        **kwargs: Additional arguments passed to BaseNemoTTSService
    """

    def __init__(
        self,
        model_path: str = "/path/to/model",
        device: str = "cuda",
        sample_rate: int = 24000,
        reference_audio_path: Optional[str] = None,
        reference_audio_text: Optional[str] = None,
        streaming: bool = True,
        use_subprocess: bool = True,  # Default to subprocess to avoid dependency conflicts
        fish_speech_repo: Optional[str] = None,  # Path to cloned fish-speech repo
        **kwargs,
    ):
        self._model_path = model_path
        self._reference_audio_path = reference_audio_path
        self._reference_audio_text = reference_audio_text or ""
        self._streaming = streaming
        self._use_subprocess = use_subprocess
        self._fish_speech_repo = fish_speech_repo
        self._codec = None
        self._llm = None
        self._reference_tokens = None
        model_name = "openaudio-s1-mini-korean"
        super().__init__(model=model_name, device=device, sample_rate=sample_rate, **kwargs)

    def _setup_model(self):
        """Initialize the OpenAudio S1-mini model."""
        import os

        logger.info(f"Loading OpenAudio S1-mini model from {self._model_path}")

        # Validate model path
        codec_path = os.path.join(self._model_path, "codec.pth")
        if not os.path.exists(codec_path):
            raise FileNotFoundError(
                f"OpenAudio S1-mini codec.pth not found at {codec_path}. "
                f"Download with: hf download fishaudio/openaudio-s1-mini "
                f"--local-dir {self._model_path}"
            )

        # Find fish-speech repository for subprocess mode
        self._fish_speech_dir = self._find_fish_speech_repo()

        # If using subprocess mode (recommended to avoid dependency conflicts)
        if self._use_subprocess:
            logger.info("Using subprocess mode for OpenAudio S1-mini (recommended for dependency isolation)")
            if self._fish_speech_dir:
                logger.info(f"Fish Speech repo found at: {self._fish_speech_dir}")
            else:
                logger.warning(
                    "Fish Speech repo not found. Clone it with: "
                    "git clone https://github.com/fishaudio/fish-speech.git "
                    "/path/to/workspace"
                )
            return {"model_path": self._model_path, "codec_path": codec_path, "mode": "subprocess"}

        # Try to load as Python module (requires pip install -e .)
        try:
            logger.info("Attempting to load Fish Speech as Python module...")
            from fish_speech.models.dac.model import DACModel
            from fish_speech.models.text2semantic.llama import LlamaModel

            # Load codec (DAC) model
            logger.info("Loading codec model...")
            self._codec = DACModel.load(codec_path, device=self._device)
            self._codec.eval()

            # Load LLM model
            logger.info("Loading LLM model...")
            self._llm = LlamaModel.load(self._model_path, device=self._device)
            self._llm.eval()

            logger.info("OpenAudio S1-mini models loaded successfully as Python modules")

            # Load reference audio if provided
            if self._reference_audio_path and os.path.exists(self._reference_audio_path):
                self._load_reference_audio()

            return {"codec": self._codec, "llm": self._llm, "mode": "python"}

        except ImportError as e:
            logger.warning(
                f"Fish Speech package not installed: {e}. "
                f"Falling back to subprocess mode."
            )
            return {"model_path": self._model_path, "codec_path": codec_path, "mode": "subprocess"}

    def _find_fish_speech_repo(self) -> Optional[str]:
        """Find the fish-speech repository for subprocess calls."""
        import os

        # Check user-specified path first
        if self._fish_speech_repo and os.path.exists(self._fish_speech_repo):
            return self._fish_speech_repo

        # Check common locations
        fish_speech_paths = [
            "/path/to/workspace",  # User's preferred location
            "/path/to/workspace",
            os.path.expanduser("~/fish-speech"),
            "/opt/fish-speech",
        ]

        for path in fish_speech_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "fish_speech")):
                return path

        return None

    def _load_reference_audio(self):
        """Load and encode reference audio for voice cloning."""
        import os

        if not self._reference_audio_path or not os.path.exists(self._reference_audio_path):
            logger.warning(f"Reference audio not found: {self._reference_audio_path}")
            return

        try:
            import torchaudio

            logger.info(f"Loading reference audio: {self._reference_audio_path}")

            # Load audio
            waveform, sr = torchaudio.load(self._reference_audio_path)

            # Resample if needed (Fish Speech expects specific sample rate)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                waveform = resampler(waveform)

            # Extract VQ tokens from reference audio using codec
            if self._codec is not None:
                with torch.no_grad():
                    self._reference_tokens = self._codec.encode(waveform.to(self._device))
                logger.info(f"Reference audio encoded. Token shape: {self._reference_tokens.shape}")
            else:
                logger.warning("Codec not loaded, cannot encode reference audio")

        except Exception as e:
            logger.error(f"Failed to load reference audio: {e}")
            self._reference_tokens = None

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Generate audio using OpenAudio S1-mini.

        OpenAudio S1-mini uses a 3-step inference process:
        1. Extract VQ tokens from reference audio (if provided)
        2. Generate semantic tokens from text using LLM
        3. Convert tokens to audio using codec

        Args:
            text: Korean text to convert to speech

        Yields:
            Audio data as numpy arrays
        """
        try:
            logger.debug(f"Generating Korean TTS with OpenAudio S1-mini for text: [{text}]")

            # Use subprocess mode if configured (recommended for dependency isolation)
            if self._use_subprocess or self._codec is None or self._llm is None:
                logger.debug("Using subprocess mode for audio generation")
                yield from self._generate_audio_subprocess(text)
                return

            with torch.no_grad():
                # Step 1: Generate semantic tokens from text
                # The LLM generates semantic tokens conditioned on reference if available
                if self._reference_tokens is not None and self._reference_audio_text:
                    # Zero-shot voice cloning mode
                    semantic_tokens = self._llm.generate(
                        text=text,
                        prompt_text=self._reference_audio_text,
                        prompt_tokens=self._reference_tokens,
                    )
                else:
                    # Random voice mode
                    semantic_tokens = self._llm.generate(text=text)

                # Step 2: Convert semantic tokens to audio using codec
                audio = self._codec.decode(semantic_tokens)

                # Convert to numpy
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()

                # Flatten if needed
                if audio.ndim > 1:
                    audio = audio.squeeze()

                # Chunk for streaming
                chunk_size = 4096
                for i in range(0, len(audio), chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) > 0:
                        yield chunk

        except Exception as e:
            logger.error(f"Error generating audio with OpenAudio S1-mini: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_audio_subprocess(self, text: str) -> Iterator[np.ndarray]:
        """Generate audio using subprocess calls to OpenAudio/Fish Speech CLI.

        This is the recommended mode to avoid dependency conflicts with the
        existing Korean Voice Agent environment. It calls the fish-speech
        CLI scripts as external processes.
        """
        import subprocess
        import tempfile
        import os
        import sys

        try:
            # Create temp directory for intermediate files
            # Fish Speech CLI outputs to fixed filenames in the working directory:
            # - dac/inference.py (encode): outputs "fake.npy"
            # - text2semantic/inference.py: outputs "codes_0.npy"
            # - dac/inference.py (decode): outputs "fake.wav"
            with tempfile.TemporaryDirectory() as tmpdir:
                # Use pre-found fish-speech directory or search again
                fish_speech_dir = self._fish_speech_dir or self._find_fish_speech_repo()

                if fish_speech_dir is None:
                    raise FileNotFoundError(
                        "Fish Speech repository not found. "
                        "Clone it with: git clone https://github.com/fishaudio/fish-speech.git "
                        "/path/to/workspace"
                    )

                # Set up environment with PYTHONPATH pointing to fish-speech repo
                # This is required because we don't run 'pip install -e .' to avoid dependency conflicts
                env = os.environ.copy()
                existing_pythonpath = env.get("PYTHONPATH", "")
                if existing_pythonpath:
                    env["PYTHONPATH"] = f"{fish_speech_dir}:{existing_pythonpath}"
                else:
                    env["PYTHONPATH"] = fish_speech_dir

                logger.debug(f"Using PYTHONPATH: {env['PYTHONPATH']}")

                codec_path = os.path.join(self._model_path, "codec.pth")

                # Step 1: Extract reference tokens (if reference audio provided)
                prompt_args = []
                if self._reference_audio_path and os.path.exists(self._reference_audio_path):
                    # Extract VQ tokens from reference (outputs fake.npy in cwd)
                    subprocess.run([
                        sys.executable, f"{fish_speech_dir}/fish_speech/models/dac/inference.py",
                        "-i", self._reference_audio_path,
                        "--checkpoint-path", codec_path,
                    ], check=True, cwd=tmpdir, env=env)

                    ref_tokens_file = os.path.join(tmpdir, "fake.npy")
                    if os.path.exists(ref_tokens_file):
                        prompt_args = [
                            "--prompt-text", self._reference_audio_text,
                            "--prompt-tokens", ref_tokens_file,
                        ]
                    else:
                        logger.warning(f"Reference tokens file not created: {ref_tokens_file}")

                # Step 2: Generate semantic tokens from text
                # Outputs codes_0.npy in cwd (no -o option available)
                subprocess.run([
                    sys.executable, f"{fish_speech_dir}/fish_speech/models/text2semantic/inference.py",
                    "--text", text,
                    "--checkpoint-path", self._model_path,
                    "--compile",
                ] + prompt_args, check=True, cwd=tmpdir, env=env)

                codes_file = os.path.join(tmpdir, "codes_0.npy")
                if not os.path.exists(codes_file):
                    logger.error(f"Codes file not created: {codes_file}")
                    logger.error(f"Files in tmpdir: {os.listdir(tmpdir)}")
                    return

                # Step 3: Convert tokens to audio
                # Outputs fake.wav in cwd (no -o option available)
                subprocess.run([
                    sys.executable, f"{fish_speech_dir}/fish_speech/models/dac/inference.py",
                    "-i", codes_file,
                    "--checkpoint-path", codec_path,
                ], check=True, cwd=tmpdir, env=env)

                # Fish Speech outputs to fake.wav in the working directory
                output_file = os.path.join(tmpdir, "fake.wav")

                # Load and yield audio
                if os.path.exists(output_file):
                    import torchaudio
                    waveform, sr = torchaudio.load(output_file)
                    audio = waveform.numpy().squeeze()

                    # Chunk for streaming
                    chunk_size = 4096
                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i:i + chunk_size]
                        if len(chunk) > 0:
                            yield chunk
                else:
                    logger.error(f"Output file not created: {output_file}")
                    logger.error(f"Files in tmpdir: {os.listdir(tmpdir)}")

        except subprocess.CalledProcessError as e:
            logger.error(f"OpenAudio S1-mini subprocess error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in OpenAudio S1-mini subprocess fallback: {e}")
            raise


class FishSpeechAPIService(BaseNemoTTSService):
    """Text-to-Speech service using Fish Speech HTTP API for low-latency streaming.

    This service connects to a locally running Fish Speech API server.
    The API server keeps the model in GPU memory for fast inference (~1-3 seconds).

    This is the RECOMMENDED approach for real-time streaming voice agents.

    **Voice Cloning Support:**
    - Automatically registers reference audio on startup via /v1/references/add
    - Uses reference_id for consistent voice across all TTS requests
    - Supports memory caching for faster inference

    Prerequisites:
    1. Start the Fish Speech API server:
       cd /path/to/workspace
       export PYTHONPATH="/path/to/workspace"
       python -m tools.api_server \\
           --listen 0.0.0.0:8080 \\
           --llama-checkpoint-path "/path/to/model" \\
           --decoder-checkpoint-path "/path/to/model" \\
           --decoder-config-name modded_dac_vq \\
           --compile

    2. Configure the Voice Agent to use this service:
       tts:
         type: fish_speech_api
         api_url: "http://localhost:8080"
         reference_audio_path: "/path/to/korean_voice.wav"
         reference_audio_text: "참조 오디오의 정확한 transcript"

    Args:
        api_url: URL of the Fish Speech API server (default: http://localhost:8080)
        device: Device string (not used, API server handles device)
        sample_rate: Audio sample rate in Hz (default: 44100, Fish Speech native rate)
        reference_audio_path: Path to reference audio for voice cloning (10-30s)
        reference_audio_text: Transcript of the reference audio
        reference_id: Optional pre-registered reference ID (if already registered)
        use_memory_cache: Enable memory caching for faster inference (default: "on")
        timeout: Request timeout in seconds (default: 60)
        **kwargs: Additional arguments passed to BaseNemoTTSService

    Note:
        Fish Speech API (OpenAudio S1-mini) outputs audio at 44100Hz natively.
        Setting sample_rate to 44100 preserves the original quality without resampling.
        For lower bandwidth, you can set sample_rate to 24000 or 16000, but this will
        involve resampling which may slightly affect audio quality.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        device: str = "cuda",
        sample_rate: int = 44100,  # Fish Speech API native rate for best quality
        reference_audio_path: Optional[str] = None,
        reference_audio_text: Optional[str] = None,
        reference_id: Optional[str] = None,
        use_memory_cache: str = "on",
        timeout: int = 60,
        **kwargs,
    ):
        self._api_url = api_url.rstrip("/")
        self._reference_audio_path = reference_audio_path
        self._reference_audio_text = reference_audio_text or ""
        self._reference_id = reference_id  # Can be pre-set or auto-registered
        self._use_memory_cache = use_memory_cache
        self._timeout = timeout
        self._session = None
        self._aiohttp_session: Optional[aiohttp.ClientSession] = None if AIOHTTP_AVAILABLE else None
        self._reference_registered = False  # Track if reference was registered
        self._use_streaming = True  # Enable streaming by default
        self._interrupted = False  # Track interruption state
        model_name = "fish-speech-api"
        super().__init__(model=model_name, device=device, sample_rate=sample_rate, **kwargs)

    def _setup_model(self):
        """Initialize HTTP session, verify API server, and register reference audio."""
        import requests
        import os
        import hashlib

        logger.info(f"Connecting to Fish Speech API server at {self._api_url}")

        # Create a session for connection pooling
        self._session = requests.Session()

        # Test connection to API server
        server_ready = False
        try:
            for endpoint in ["/", "/health", "/v1/health", "/docs"]:
                try:
                    response = self._session.get(
                        f"{self._api_url}{endpoint}",
                        timeout=5
                    )
                    if response.status_code == 200:
                        logger.info(f"Fish Speech API server is running (endpoint: {endpoint})")
                        server_ready = True
                        break
                except:
                    continue

            if not server_ready:
                logger.warning(
                    f"Fish Speech API server health check failed at {self._api_url}. "
                    f"Make sure the server is running."
                )

        except requests.exceptions.ConnectionError as e:
            logger.error(
                f"Cannot connect to Fish Speech API server at {self._api_url}. "
                f"Please start the server first. Error: {e}"
            )
            raise ConnectionError(
                f"Fish Speech API server not running at {self._api_url}. "
                f"Start it with: ./start_fish_speech_server.sh"
            )

        # Register reference audio if provided and not already registered
        if self._reference_audio_path and not self._reference_id:
            self._register_reference_audio()

        # If reference_id is set from config (not auto-registered), verify it exists
        # Skip verification if we just registered successfully - trust the registration result
        if self._reference_id and not self._reference_registered:
            self._verify_reference_exists()

        return {
            "mode": "api",
            "url": self._api_url,
            "reference_id": self._reference_id,
            "reference_registered": self._reference_registered
        }

    def _register_reference_audio(self):
        """Register reference audio with Fish Speech API server.

        Calls /v1/references/add to upload and register the reference audio.
        The registered reference_id will be used for all subsequent TTS requests.
        """
        import os
        import hashlib

        if not self._reference_audio_path:
            return

        if not os.path.exists(self._reference_audio_path):
            logger.warning(f"Reference audio file not found: {self._reference_audio_path}")
            return

        # Generate a unique reference_id based on file content hash
        # This ensures consistent ID for the same audio file
        with open(self._reference_audio_path, "rb") as f:
            audio_content = f.read()
            file_hash = hashlib.md5(audio_content).hexdigest()[:12]

        # Create a human-readable reference_id
        # Fish Speech API only allows: alphanumeric, hyphens, underscores, and spaces
        base_name = os.path.splitext(os.path.basename(self._reference_audio_path))[0]
        # Sanitize: replace dots and other invalid characters with underscore
        import re
        sanitized_name = re.sub(r'[^a-zA-Z0-9\-_ ]', '_', base_name)
        reference_id = f"voice_agent_{sanitized_name}_{file_hash}"

        logger.info(f"Registering reference audio: {self._reference_audio_path}")
        logger.info(f"Reference ID: {reference_id}")
        logger.info(f"Reference text: {self._reference_audio_text[:50]}..." if len(self._reference_audio_text) > 50 else f"Reference text: {self._reference_audio_text}")

        try:
            # First, check if reference already exists
            ref_ids = []
            try:
                list_response = self._session.get(
                    f"{self._api_url}/v1/references/list",
                    timeout=10
                )

                if list_response.status_code == 200:
                    # Handle empty response gracefully - Fish Speech API may return empty body
                    response_text = list_response.text.strip()
                    if response_text:
                        try:
                            existing_refs = list_response.json()
                            # Handle different response formats
                            if isinstance(existing_refs, list):
                                ref_ids = existing_refs
                            elif isinstance(existing_refs, dict):
                                ref_ids = existing_refs.get("references", existing_refs.get("ids", []))
                        except Exception as json_err:
                            # This is expected for some Fish Speech API versions - log at debug level
                            logger.debug(f"References list response not JSON parseable: {json_err}")
                    else:
                        logger.debug("Empty response from /v1/references/list - no references registered yet")
            except Exception as list_err:
                logger.debug(f"Could not fetch references list: {list_err}")

            if reference_id in ref_ids:
                logger.info(f"Reference '{reference_id}' already exists, reusing it")
                self._reference_id = reference_id
                self._reference_registered = True
                return

            # Register new reference via /v1/references/add
            with open(self._reference_audio_path, "rb") as audio_file:
                files = {
                    "audio": (os.path.basename(self._reference_audio_path), audio_file, "audio/wav")
                }
                data = {
                    "id": reference_id,
                    "text": self._reference_audio_text
                }

                response = self._session.post(
                    f"{self._api_url}/v1/references/add",
                    files=files,
                    data=data,
                    timeout=60
                )

            if response.status_code == 200:
                logger.info(f"Successfully registered reference audio with ID: {reference_id}")
                self._reference_id = reference_id
                self._reference_registered = True
            elif response.status_code == 409:
                # Reference already exists (race condition or previous registration)
                logger.info(f"Reference '{reference_id}' already exists (409), reusing it")
                self._reference_id = reference_id
                self._reference_registered = True
            else:
                logger.error(f"Failed to register reference audio: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}")
                # Fall back to in-context mode (references array)
                logger.warning("Falling back to in-context reference mode (less efficient)")

        except Exception as e:
            logger.error(f"Error registering reference audio: {e}")
            import traceback
            traceback.print_exc()

    def _verify_reference_exists(self):
        """Verify that the configured reference_id exists on the server."""
        if not self._reference_id:
            return

        try:
            response = self._session.get(
                f"{self._api_url}/v1/references/list",
                timeout=10
            )

            ref_ids = []
            if response.status_code == 200:
                # Handle empty response gracefully - Fish Speech API may return empty body
                response_text = response.text.strip()
                if response_text:
                    try:
                        existing_refs = response.json()
                        if isinstance(existing_refs, list):
                            ref_ids = existing_refs
                        elif isinstance(existing_refs, dict):
                            ref_ids = existing_refs.get("references", existing_refs.get("ids", []))
                    except Exception as json_err:
                        # This is expected for some Fish Speech API versions
                        logger.debug(f"References list response not JSON parseable: {json_err}")
                else:
                    logger.debug("Empty response from /v1/references/list")

            if self._reference_id in ref_ids:
                logger.info(f"Reference '{self._reference_id}' verified on server")
                self._reference_registered = True
            else:
                logger.warning(f"Reference '{self._reference_id}' not found on server")
                if ref_ids:
                    logger.warning(f"Available references: {ref_ids}")
                # If we have audio path, try to register
                if self._reference_audio_path:
                    logger.info("Attempting to register reference audio...")
                    self._reference_id = None  # Reset to trigger registration
                    self._register_reference_audio()

        except Exception as e:
            logger.warning(f"Could not verify reference existence: {e}")

    def _generate_audio(self, text: str) -> Iterator[np.ndarray]:
        """Generate audio using Fish Speech HTTP API.

        Uses reference_id if registered, otherwise falls back to in-context references.

        Args:
            text: Korean text to convert to speech

        Yields:
            Audio data as numpy arrays
        """
        import requests
        import io
        import base64
        import os

        try:
            logger.debug(f"Generating TTS via API for text: [{text}]")

            # Build request payload according to Fish Speech API spec
            payload = {
                "text": text,
                "format": "wav",
                "use_memory_cache": self._use_memory_cache,
            }

            # Use registered reference_id if available (most efficient)
            if self._reference_id and self._reference_registered:
                payload["reference_id"] = self._reference_id
                logger.debug(f"Using registered reference_id: {self._reference_id}")
            # Fall back to in-context references if we have audio but no registration
            elif self._reference_audio_path and os.path.exists(self._reference_audio_path):
                logger.debug("Using in-context reference (less efficient)")
                with open(self._reference_audio_path, "rb") as f:
                    audio_bytes = f.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

                payload["references"] = [{
                    "audio": audio_base64,
                    "text": self._reference_audio_text
                }]

            # Try the primary endpoint
            url = f"{self._api_url}/v1/tts"
            response = None

            try:
                logger.debug(f"Sending TTS request to {url}")
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=self._timeout
                )

                if response.status_code != 200:
                    # Try alternative endpoints
                    for alt_endpoint in ["/api/tts", "/tts"]:
                        alt_url = f"{self._api_url}{alt_endpoint}"
                        try:
                            response = self._session.post(
                                alt_url,
                                json=payload,
                                timeout=self._timeout
                            )
                            if response.status_code == 200:
                                break
                        except:
                            continue

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error to Fish Speech API: {e}")
                return
            except Exception as e:
                logger.warning(f"API request error: {e}")

            if response is None or response.status_code != 200:
                error_msg = response.text[:500] if response else "No response"
                status_code = response.status_code if response else "N/A"
                logger.error(f"Fish Speech API error (status={status_code}): {error_msg}")
                return

            # Process audio response
            content_type = response.headers.get("content-type", "")

            if "audio" in content_type or len(response.content) > 1000:
                # Load audio from response
                import torchaudio

                audio_buffer = io.BytesIO(response.content)

                try:
                    waveform, sr = torchaudio.load(audio_buffer)
                    audio = waveform.numpy().squeeze()

                    # Resample if needed
                    if sr != self._sample_rate:
                        import torchaudio.transforms as T
                        resampler = T.Resample(sr, self._sample_rate)
                        waveform = resampler(torch.from_numpy(audio).unsqueeze(0))
                        audio = waveform.numpy().squeeze()

                    # Chunk for streaming
                    chunk_size = 4096
                    for i in range(0, len(audio), chunk_size):
                        chunk = audio[i:i + chunk_size]
                        if len(chunk) > 0:
                            yield chunk

                except Exception as e:
                    logger.error(f"Error loading audio from API response: {e}")
                    return
            else:
                logger.error(f"API response is not audio. Content-Type: {content_type}")
                logger.error(f"Response: {response.text[:500]}")

        except requests.exceptions.Timeout:
            logger.error(f"Fish Speech API request timed out after {self._timeout}s")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Fish Speech API connection error: {e}")
        except Exception as e:
            logger.error(f"Error calling Fish Speech API: {e}")
            import traceback
            traceback.print_exc()

    async def _get_aiohttp_session(self) -> "aiohttp.ClientSession":
        """Get or create aiohttp session for async HTTP requests."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is not available. Install with: pip install aiohttp")

        if self._aiohttp_session is None or self._aiohttp_session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._aiohttp_session = aiohttp.ClientSession(timeout=timeout)
        return self._aiohttp_session

    async def _async_generate_audio_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio using Fish Speech HTTP API with true streaming.

        Uses aiohttp to stream audio chunks as they are generated by the server,
        avoiding the need to wait for the complete audio before starting playback.

        Args:
            text: Korean text to convert to speech

        Yields:
            Audio data as bytes (WAV format chunks)
        """
        import base64
        import os

        try:
            logger.debug(f"[Streaming TTS] Generating for text: [{text}]")

            # Build request payload with streaming enabled
            payload = {
                "text": text,
                "format": "wav",  # Streaming only supports WAV
                "streaming": True,  # Enable streaming mode
                "use_memory_cache": self._use_memory_cache,
            }

            # Use registered reference_id if available
            if self._reference_id and self._reference_registered:
                payload["reference_id"] = self._reference_id
                logger.debug(f"[Streaming TTS] Using reference_id: {self._reference_id}")
            elif self._reference_audio_path and os.path.exists(self._reference_audio_path):
                logger.debug("[Streaming TTS] Using in-context reference")
                with open(self._reference_audio_path, "rb") as f:
                    audio_bytes = f.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                payload["references"] = [{
                    "audio": audio_base64,
                    "text": self._reference_audio_text
                }]

            url = f"{self._api_url}/v1/tts"
            session = await self._get_aiohttp_session()

            logger.debug(f"[Streaming TTS] Sending request to {url}")

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"[Streaming TTS] API error (status={response.status}): {error_text[:500]}")
                    return

                content_type = response.headers.get("content-type", "")
                logger.debug(f"[Streaming TTS] Response content-type: {content_type}")

                # Stream audio chunks as they arrive
                chunk_count = 0
                async for chunk in response.content.iter_chunked(8192):
                    if self._interrupted:
                        logger.info("[Streaming TTS] Interrupted, stopping stream")
                        break

                    if chunk:
                        chunk_count += 1
                        if chunk_count == 1:
                            logger.debug(f"[Streaming TTS] First chunk received ({len(chunk)} bytes)")
                        yield chunk

                logger.debug(f"[Streaming TTS] Completed, total chunks: {chunk_count}")

        except aiohttp.ClientError as e:
            logger.error(f"[Streaming TTS] aiohttp error: {e}")
        except Exception as e:
            logger.error(f"[Streaming TTS] Error: {e}")
            import traceback
            traceback.print_exc()

    def set_interrupted(self, interrupted: bool = True):
        """Set the interruption state. Called when user starts speaking."""
        self._interrupted = interrupted
        if interrupted:
            logger.debug("[TTS] Interruption flag set")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames, handling interruption signals.

        When a StartInterruptionFrame is received, sets the interruption flag
        to stop ongoing TTS generation.
        """
        # Handle interruption signals
        if isinstance(frame, StartInterruptionFrame):
            logger.info("[FishSpeechAPIService] Received interruption signal")
            self.set_interrupted(True)
            # Pass the frame downstream
            await self.push_frame(frame, direction)
            return

        # Reset interruption flag when starting new TTS (handled in run_tts)

        # Call parent implementation for standard frame processing
        await super().process_frame(frame, direction)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using streaming Fish Speech API.

        This override uses async streaming to deliver audio chunks immediately
        as they are generated, reducing first-token latency significantly.
        """
        text = self._handle_think_tokens(text)

        if not text:
            yield None
            return

        logger.debug(f"{self}: Generating TTS [{text}]")
        self._interrupted = False

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame()

            # Use streaming mode if aiohttp is available
            if AIOHTTP_AVAILABLE and self._use_streaming:
                async for frame in self._process_streaming_audio(text):
                    if self._interrupted:
                        logger.info(f"{self}: TTS interrupted by user")
                        break
                    yield frame
            else:
                # Fallback to original synchronous method
                logger.debug(f"{self}: Using fallback non-streaming TTS")
                async for frame in self._run_tts_fallback(text):
                    yield frame

            yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            yield ErrorFrame(error=f"TTS generation error: {str(e)}")

    async def _process_streaming_audio(self, text: str) -> AsyncGenerator[Frame, None]:
        """Process streaming audio chunks and yield TTS frames.

        Streams audio directly from the API response without waiting for
        complete audio generation. Includes resampling if source sample rate
        differs from configured output sample rate.
        """
        import struct
        import torchaudio.transforms as T

        first_chunk = True
        audio_buffer = io.BytesIO()
        wav_header_parsed = False
        source_sample_rate = self._sample_rate  # Will be updated from WAV header
        bytes_per_sample = 2  # 16-bit PCM
        num_channels = 1
        resampler = None

        # Accumulate raw PCM for resampling in larger chunks
        pcm_accumulator = io.BytesIO()
        MIN_RESAMPLE_SAMPLES = 4096  # Minimum samples to accumulate before resampling

        async for chunk in self._async_generate_audio_streaming(text):
            if self._interrupted:
                break

            audio_buffer.write(chunk)

            # Parse WAV header from first chunk to get actual sample rate
            if not wav_header_parsed and audio_buffer.tell() >= 44:
                audio_buffer.seek(0)
                header = audio_buffer.read(44)

                # Verify WAV header (RIFF....WAVEfmt )
                if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                    # Extract audio parameters from WAV header
                    num_channels = struct.unpack('<H', header[22:24])[0]
                    source_sample_rate = struct.unpack('<I', header[24:28])[0]
                    bits_per_sample = struct.unpack('<H', header[34:36])[0]
                    bytes_per_sample = bits_per_sample // 8

                    logger.debug(
                        f"[Streaming TTS] WAV: {source_sample_rate}Hz, "
                        f"{bits_per_sample}-bit, {num_channels}ch"
                    )

                    # Create resampler if source rate differs from target
                    if source_sample_rate != self._sample_rate:
                        resampler = T.Resample(
                            orig_freq=source_sample_rate,
                            new_freq=self._sample_rate
                        )
                        logger.debug(
                            f"[Streaming TTS] Resampling {source_sample_rate}Hz -> {self._sample_rate}Hz"
                        )

                    wav_header_parsed = True

                    # Read remaining data after header
                    remaining = audio_buffer.read()
                    audio_buffer = io.BytesIO()
                    audio_buffer.write(remaining)
                else:
                    wav_header_parsed = True  # Not a valid WAV, process as raw
                    audio_buffer.seek(0, 2)  # Seek to end

            # Process accumulated audio data
            if wav_header_parsed:
                audio_buffer.seek(0)
                audio_data = audio_buffer.read()

                # Keep some buffer for incomplete samples
                usable_len = (len(audio_data) // bytes_per_sample) * bytes_per_sample

                if usable_len > 0:
                    pcm_accumulator.write(audio_data[:usable_len])

                    # Keep remaining bytes for next iteration
                    remaining = audio_data[usable_len:]
                    audio_buffer = io.BytesIO()
                    audio_buffer.write(remaining)

                    # Check if we have enough samples to process
                    pcm_accumulator.seek(0, 2)  # Seek to end
                    accumulated_bytes = pcm_accumulator.tell()
                    accumulated_samples = accumulated_bytes // bytes_per_sample

                    if accumulated_samples >= MIN_RESAMPLE_SAMPLES:
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False
                            logger.debug(f"[Streaming TTS] First audio frame at {accumulated_samples} samples")

                        # Process accumulated PCM data
                        pcm_accumulator.seek(0)
                        pcm_bytes = pcm_accumulator.read()
                        pcm_accumulator = io.BytesIO()

                        # Convert bytes to numpy array
                        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                        # Handle stereo -> mono if needed
                        if num_channels == 2:
                            audio_np = audio_np.reshape(-1, 2).mean(axis=1)

                        # Resample if needed
                        if resampler is not None:
                            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                            audio_resampled = resampler(audio_tensor)
                            audio_np = audio_resampled.squeeze(0).numpy()

                        # Convert back to int16 bytes
                        audio_int16 = (audio_np * 32767).astype(np.int16)
                        audio_bytes = audio_int16.tobytes()

                        # Yield audio chunks
                        for i in range(0, len(audio_bytes), self.chunk_size):
                            audio_chunk = audio_bytes[i:i + self.chunk_size]
                            if audio_chunk:
                                frame = TTSAudioRawFrame(
                                    audio=audio_chunk,
                                    sample_rate=self._sample_rate,
                                    num_channels=1
                                )
                                yield frame
                else:
                    audio_buffer.seek(0, 2)  # Seek to end, keep accumulating

        # Flush remaining audio
        if wav_header_parsed:
            # Process any remaining PCM data
            pcm_accumulator.seek(0, 2)
            if pcm_accumulator.tell() > 0:
                if first_chunk:
                    await self.stop_ttfb_metrics()

                pcm_accumulator.seek(0)
                pcm_bytes = pcm_accumulator.read()

                if len(pcm_bytes) >= bytes_per_sample:
                    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    if num_channels == 2:
                        audio_np = audio_np.reshape(-1, 2).mean(axis=1)

                    if resampler is not None:
                        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                        audio_resampled = resampler(audio_tensor)
                        audio_np = audio_resampled.squeeze(0).numpy()

                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()

                    for i in range(0, len(audio_bytes), self.chunk_size):
                        audio_chunk = audio_bytes[i:i + self.chunk_size]
                        if audio_chunk:
                            frame = TTSAudioRawFrame(
                                audio=audio_chunk,
                                sample_rate=self._sample_rate,
                                num_channels=1
                            )
                            yield frame

    async def _run_tts_fallback(self, text: str) -> AsyncGenerator[Frame, None]:
        """Fallback TTS method using the original synchronous approach."""
        import uuid

        request_id = str(uuid.uuid4())
        request_queue = asyncio.Queue()
        self._pending_requests[request_id] = request_queue

        try:
            await self._tts_queue.put((text, request_id))
            result = await request_queue.get()
            status, data = result

            if status == 'error':
                logger.error(f"{self} TTS generation error: {data}")
                yield ErrorFrame(error=f"TTS generation error: {str(data)}")
                return

            audio_result = data
            if audio_result is None:
                logger.error(f"{self} TTS model returned None")
                yield ErrorFrame(error="TTS generation failed - no audio returned")
                return

            await self.start_tts_usage_metrics(text)

            if inspect.isgenerator(audio_result) or (hasattr(audio_result, '__iter__') and hasattr(audio_result, '__next__')):
                first_chunk = True
                for audio_chunk in audio_result:
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False

                    if audio_chunk is None:
                        break

                    audio_bytes = self._convert_to_bytes(audio_chunk)
                    chunk_size = self.chunk_size
                    for i in range(0, len(audio_bytes), chunk_size):
                        audio_chunk_bytes = audio_bytes[i:i + chunk_size]
                        if audio_chunk_bytes:
                            yield TTSAudioRawFrame(
                                audio=audio_chunk_bytes,
                                sample_rate=self.sample_rate,
                                num_channels=1
                            )
            else:
                await self.stop_ttfb_metrics()
                audio_bytes = self._convert_to_bytes(audio_result)
                chunk_size = self.chunk_size
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    if chunk:
                        yield TTSAudioRawFrame(
                            audio=chunk,
                            sample_rate=self.sample_rate,
                            num_channels=1
                        )
        finally:
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]

    async def cleanup(self):
        """Cleanup resources including aiohttp session."""
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()
            self._aiohttp_session = None
        logger.debug("[FishSpeechAPIService] Cleanup completed")
