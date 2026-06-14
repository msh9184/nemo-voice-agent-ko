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
# NOTE: This file will be deprecated in the future, as the new inference pipeline will replace it.

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from omegaconf import open_dict
from loguru import logger

import nemo.collections.asr as nemo_asr
from nemo.agents.voice_agent.pipecat.services.nemo.utils import CacheFeatureBufferer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


@dataclass
class ASRResult:
    text: str
    is_final: bool
    eou_prob: Optional[float] = None
    eob_prob: Optional[float] = None
    eou_latency: Optional[float] = None
    eob_latency: Optional[float] = None
    processing_time: Optional[float] = None


class NemoStreamingASRService:
    def __init__(
        self,
        model: str = "nvidia/parakeet_realtime_eou_120m-v1",
        att_context_size: List[int] = [70, 1],
        device: str = "cuda",
        eou_string: str = "<EOU>",
        eob_string: str = "<EOB>",
        decoder_type: str = None,
        chunk_size: int = -1,
        shift_size: int = -1,
        left_chunks: int = 2,
        sample_rate: int = 16000,
        frame_len_in_secs: float = 0.08,
        use_amp: bool = False,
        chunk_size_in_secs: float = 0.08,
    ):
        self.model = model
        self.eou_string = eou_string
        self.eob_string = eob_string
        self.device = device
        self.att_context_size = att_context_size
        self.decoder_type = decoder_type
        self.chunk_size = chunk_size
        self.shift_size = shift_size
        self.left_chunks = left_chunks
        self.asr_model = self._load_model(model)
        self.tokenizer: SentencePieceTokenizer = self.asr_model.tokenizer
        self.use_amp = use_amp
        self.pad_and_drop_preencoded = False
        self.blank_id = self.get_blank_id()
        self.chunk_size_in_secs = chunk_size_in_secs

        assert len(self.att_context_size) == 2, "Att context size must be a list of two integers"
        assert (
            self.att_context_size[0] >= 0
        ), f"Left att context size must be greater than 0: {self.att_context_size[0]}"
        assert (
            self.att_context_size[1] >= 0
        ), f"Right att context size must be greater than 0: {self.att_context_size[1]}"

        window_stride_in_secs = self.asr_model.cfg.preprocessor.window_stride
        model_stride = self.asr_model.cfg.encoder.subsampling_factor
        self.model_chunk_size = self.asr_model.encoder.streaming_cfg.chunk_size
        if isinstance(self.model_chunk_size, list):
            self.model_chunk_size = self.model_chunk_size[1]
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(self.pre_encode_cache_size, list):
            self.pre_encode_cache_size = self.pre_encode_cache_size[1]
        self.pre_encode_cache_size_in_secs = self.pre_encode_cache_size * window_stride_in_secs

        self.tokens_per_frame = math.ceil(np.trunc(self.chunk_size_in_secs / window_stride_in_secs) / model_stride)

        # Calculate proper left_chunks based on att_context_size for correct cache size
        # In chunked_limited style: chunk_size_for_att = right_context + 1
        # left_chunks must provide enough cache to cover left_context frames
        chunk_size_for_streaming = self.model_chunk_size // model_stride
        if self.att_context_size[1] >= 0:
            chunk_size_for_att = self.att_context_size[1] + 1
            if self.att_context_size[0] >= 0:
                # Calculate left_chunks to cover left_context
                calculated_left_chunks = self.att_context_size[0] // chunk_size_for_att
            else:
                calculated_left_chunks = self.left_chunks
        else:
            calculated_left_chunks = self.left_chunks

        logger.info(f"[ASR] Streaming params calculation:")
        logger.info(f"[ASR]   chunk_size_for_streaming={chunk_size_for_streaming}")
        logger.info(f"[ASR]   att_context_size={self.att_context_size}")
        logger.info(f"[ASR]   calculated_left_chunks={calculated_left_chunks} (was: {self.left_chunks})")
        logger.info(f"[ASR]   expected last_channel_cache_size={calculated_left_chunks * chunk_size_for_streaming}")

        # overwrite the encoder streaming params with proper shift size for cache aware streaming
        self.asr_model.encoder.setup_streaming_params(
            chunk_size=chunk_size_for_streaming,
            shift_size=self.tokens_per_frame,
            left_chunks=calculated_left_chunks
        )

        model_chunk_size_in_secs = self.model_chunk_size * window_stride_in_secs

        self.buffer_size_in_secs = self.pre_encode_cache_size_in_secs + model_chunk_size_in_secs

        self._audio_buffer = CacheFeatureBufferer(
            sample_rate=sample_rate,
            buffer_size_in_secs=self.buffer_size_in_secs,
            chunk_size_in_secs=self.chunk_size_in_secs,
            preprocessor_cfg=self.asr_model.cfg.preprocessor,
            device=self.device,
        )
        self._reset_cache()
        self._previous_hypotheses = self._get_blank_hypothesis()
        self._last_transcript_timestamp = time.time()
        self._transcribe_call_count = 0  # Counter for debugging

        # Stateless decoder detection and manual token accumulation
        # For Stateless Transducer decoders, dec_state is always None, so we need to
        # manually accumulate tokens across chunks since the decoder resets y_sequence
        self._is_stateless_decoder = self._detect_stateless_decoder()
        self._accumulated_tokens: List[int] = []  # Manual token accumulation for Stateless decoders
        logger.info(f"[ASR]   is_stateless_decoder={self._is_stateless_decoder}")

        # Session timeout auto-clear mechanism (safety net for stale tokens)
        # If no audio is processed for this many seconds, clear accumulated state
        # This prevents stale tokens from previous sessions contaminating new sessions
        self._session_idle_timeout_secs = 30.0  # Clear state after 30s of inactivity
        self._last_transcribe_time = time.time()

        # Log initialization parameters
        logger.info(f"[ASR] NemoStreamingASRService initialized:")
        logger.info(f"[ASR]   model={model}, device={self.device}")
        logger.info(f"[ASR]   decoder_type={self.decoder_type}")
        logger.info(f"[ASR]   buffer_size_in_secs={self.buffer_size_in_secs:.3f}s")
        logger.info(f"[ASR]   chunk_size_in_secs={self.chunk_size_in_secs:.3f}s")
        logger.info(f"[ASR]   pre_encode_cache_size_in_secs={self.pre_encode_cache_size_in_secs:.3f}s")
        logger.info(f"[ASR]   model_chunk_size={self.model_chunk_size}")
        logger.info(f"[ASR]   tokens_per_frame={self.tokens_per_frame}")
        logger.info(f"[ASR]   att_context_size={self.att_context_size}")

        # Log encoder parameters for debugging tensor shape issues
        logger.info(f"[ASR]   conv_context_size={self.asr_model.encoder.conv_context_size}")
        logger.info(f"[ASR]   n_layers={self.asr_model.encoder.n_layers}")
        logger.info(f"[ASR]   d_model={self.asr_model.encoder.d_model}")
        # n_heads is stored in _cfg, not as direct attribute
        n_heads = getattr(self.asr_model.encoder, '_cfg', {}).get('n_heads', 'N/A') if hasattr(self.asr_model.encoder, '_cfg') else 'N/A'
        logger.info(f"[ASR]   n_heads={n_heads}")
        logger.info(f"[ASR]   subsampling_factor={self.asr_model.encoder.subsampling_factor}")
        logger.info(f"[ASR]   att_context_style={self.asr_model.encoder.att_context_style}")
        logger.info(f"[ASR]   att_context_size={self.asr_model.encoder.att_context_size}")

        # Log decoding configuration for debugging
        if hasattr(self.asr_model, 'decoding') and hasattr(self.asr_model.decoding, '_decoding_config'):
            dec_cfg = self.asr_model.decoding._decoding_config
            logger.info(f"[ASR]   decoding_strategy={dec_cfg.strategy if hasattr(dec_cfg, 'strategy') else 'N/A'}")
            if hasattr(dec_cfg, 'greedy'):
                logger.info(f"[ASR]   greedy.max_symbols={getattr(dec_cfg.greedy, 'max_symbols', 'N/A')}")
            logger.info(f"[ASR]   preserve_alignments={getattr(dec_cfg, 'preserve_alignments', 'N/A')}")

    def _reset_cache(self):
        (
            self._cache_last_channel,  # [17, B, 70, 512]
            self._cache_last_time,  # [17, B, 512, 8]
            self._cache_last_channel_len,  # B
        ) = self.asr_model.encoder.get_initial_cache_state(
            1
        )  # batch size is 1

    def _get_blank_hypothesis(self) -> List[Hypothesis]:
        blank_hypothesis = Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)
        return [blank_hypothesis]

    def _detect_stateless_decoder(self) -> bool:
        """Detect if the decoder is a Stateless Transducer (always returns dec_state=None).

        Stateless decoders don't maintain hidden state between chunks, so we need to
        manually accumulate tokens for streaming transcription.

        Returns:
            True if using a Stateless decoder, False otherwise.
        """
        try:
            # Check if model has a decoder and if it's stateless
            if hasattr(self.asr_model, 'decoder'):
                decoder = self.asr_model.decoder
                decoder_class_name = decoder.__class__.__name__

                # StatelessTransducerDecoder is the class name for stateless decoders
                if 'Stateless' in decoder_class_name:
                    logger.info(f"[ASR] Detected Stateless decoder: {decoder_class_name}")
                    return True

                # Also check for models that might use stateless prediction network
                if hasattr(decoder, 'blank_as_pad') or hasattr(decoder, 'stateless'):
                    if getattr(decoder, 'stateless', False):
                        logger.info(f"[ASR] Decoder has stateless=True attribute")
                        return True

            # Check model name for known stateless models
            model_name = self.model.lower() if isinstance(self.model, str) else ""
            stateless_model_patterns = ['parakeet', 'realtime', 'eou']
            if any(pattern in model_name for pattern in stateless_model_patterns):
                logger.info(f"[ASR] Model name suggests Stateless decoder: {self.model}")
                return True

            logger.info(f"[ASR] Decoder appears to be Stateful")
            return False

        except Exception as e:
            logger.warning(f"[ASR] Error detecting decoder type: {e}, assuming Stateless")
            return True  # Default to Stateless for safety

    @property
    def drop_extra_pre_encoded(self):
        return self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

    def get_blank_id(self):
        return len(self.tokenizer.vocab)

    def get_text_from_tokens(self, tokens: List[int]) -> str:
        sep = "\u2581"  # '▁'
        tokens = [int(t) for t in tokens if t != self.blank_id]
        if tokens:
            pieces = self.tokenizer.ids_to_tokens(tokens)
            text = "".join([p.replace(sep, ' ') if p.startswith(sep) else p for p in pieces])
        else:
            text = ""
        return text

    def _get_model_class_from_config(self, model_path: str):
        """
        Reads model_config.yaml from .nemo file to determine the correct model class.
        This reduces unnecessary model loading attempts and shortens loading time.

        Returns:
            Tuple of (model_class, target_string) or (None, None) if detection fails
        """
        import tarfile
        import tempfile
        import yaml
        from pathlib import Path

        # Target string to class mapping
        TARGET_TO_CLASS = {
            "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel": nemo_asr.models.EncDecHybridRNNTCTCBPEModel,
            "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel": nemo_asr.models.EncDecRNNTBPEModel,
            "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE": nemo_asr.models.EncDecCTCModelBPE,
            "nemo.collections.asr.models.hybrid_rnnt_ctc_models.EncDecHybridRNNTCTCModel": nemo_asr.models.EncDecHybridRNNTCTCModel,
            "nemo.collections.asr.models.rnnt_models.EncDecRNNTModel": nemo_asr.models.EncDecRNNTModel,
            "nemo.collections.asr.models.ctc_models.EncDecCTCModel": nemo_asr.models.EncDecCTCModel,
        }

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)

                # Extract only model_config.yaml for fast detection
                with tarfile.open(model_path, "r") as tar:
                    # Find config file
                    config_member = None
                    for member in tar.getmembers():
                        if member.name.endswith("model_config.yaml"):
                            config_member = member
                            break

                    if config_member is None:
                        logger.warning("[ASR] model_config.yaml not found in .nemo file")
                        return None, None

                    # Extract only the config file
                    tar.extract(config_member, tmpdir)
                    config_path = tmpdir / config_member.name

                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                target = config.get("target")
                if target and target in TARGET_TO_CLASS:
                    logger.info(f"[ASR] Detected model class from config: {target}")
                    return TARGET_TO_CLASS[target], target

                # Fallback: detect from config structure
                has_aux_ctc = "aux_ctc" in config
                has_joint = "joint" in config
                has_tokenizer = "tokenizer" in config

                if has_aux_ctc and has_joint:  # Hybrid
                    if has_tokenizer:
                        return nemo_asr.models.EncDecHybridRNNTCTCBPEModel, "detected:hybrid_bpe"
                    else:
                        return nemo_asr.models.EncDecHybridRNNTCTCModel, "detected:hybrid_char"
                elif has_joint:  # RNNT only
                    if has_tokenizer:
                        return nemo_asr.models.EncDecRNNTBPEModel, "detected:rnnt_bpe"
                    else:
                        return nemo_asr.models.EncDecRNNTModel, "detected:rnnt_char"
                else:  # CTC
                    if has_tokenizer:
                        return nemo_asr.models.EncDecCTCModelBPE, "detected:ctc_bpe"
                    else:
                        return nemo_asr.models.EncDecCTCModel, "detected:ctc_char"

        except Exception as e:
            logger.warning(f"[ASR] Failed to detect model class from config: {e}")
            return None, None

    def _restore_from_nemo(self, model_path: str):
        """
        Restores ASR model from .nemo file.
        First detects the correct model class from config to optimize loading time.
        """
        # Step 1: Try to detect model class from config (fast)
        detected_class, target_info = self._get_model_class_from_config(model_path)

        if detected_class is not None:
            try:
                logger.info(f"[ASR] Loading with detected class: {detected_class.__name__} ({target_info})")
                asr_model = detected_class.restore_from(model_path, map_location=torch.device(self.device))
                logger.info(f"[ASR] Successfully loaded model with {detected_class.__name__}")
                return asr_model
            except Exception as e:
                logger.warning(f"[ASR] Detected class failed: {e}. Falling back to sequential try...")

        # Step 2: Fallback - try all classes sequentially
        logger.info("[ASR] Using fallback mode: trying all model classes...")
        concrete_classes = [
            nemo_asr.models.ASRModel,  # Try default method first
            nemo_asr.models.EncDecHybridRNNTCTCBPEModel,  # FastConformer Hybrid BPE
            nemo_asr.models.EncDecRNNTBPEModel,  # RNNT BPE
            nemo_asr.models.EncDecCTCModelBPE,  # CTC BPE
            nemo_asr.models.EncDecHybridRNNTCTCModel,  # Hybrid Char
            nemo_asr.models.EncDecRNNTModel,  # RNNT Char
            nemo_asr.models.EncDecCTCModel,  # CTC Char
        ]

        last_error = None
        for model_cls in concrete_classes:
            try:
                logger.info(f"[ASR] Attempting to load model with {model_cls.__name__}...")
                asr_model = model_cls.restore_from(model_path, map_location=torch.device(self.device))
                logger.info(f"[ASR] Successfully loaded model with {model_cls.__name__}")
                return asr_model
            except TypeError as e:
                if "Can't instantiate abstract class" in str(e):
                    logger.debug(f"[ASR] {model_cls.__name__} failed (abstract class), trying next...")
                    last_error = e
                    continue
                else:
                    raise
            except Exception as e:
                # Other errors: try next class
                logger.debug(f"[ASR] {model_cls.__name__} failed: {e}, trying next...")
                last_error = e
                continue

        # All class attempts failed
        raise RuntimeError(
            f"Failed to load ASR model from {model_path}. "
            f"Last error: {last_error}\n"
            "Consider using the fix_nemo_model.py script to fix the .nemo file:\n"
            "  python examples/voice_agent/scripts/fix_nemo_model.py your_model.nemo --verbose"
        )

    def _load_model(self, model: str):
        if model.endswith(".nemo"):
            asr_model = self._restore_from_nemo(model)
        else:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(model, map_location=torch.device(self.device))

        if self.decoder_type is not None and hasattr(asr_model, "cur_decoder"):
            asr_model.change_decoding_strategy(decoder_type=self.decoder_type)
        elif isinstance(asr_model, nemo_asr.models.EncDecCTCModel):
            self.decoder_type = "ctc"
        elif isinstance(asr_model, nemo_asr.models.EncDecRNNTModel):
            self.decoder_type = "rnnt"
        else:
            raise ValueError("Decoder type not supported for this model.")

        if self.att_context_size is not None:
            if hasattr(asr_model.encoder, "set_default_att_context_size"):
                asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)
            else:
                raise ValueError("Model does not support multiple lookaheads.")
        else:
            self.att_context_size = asr_model.cfg.encoder.att_context_size

        decoding_cfg = asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.compute_timestamps = False
            decoding_cfg.preserve_alignments = True
            if hasattr(asr_model, 'joint'):  # if an RNNT model
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
            asr_model.change_decoding_strategy(decoding_cfg)

        if hasattr(asr_model.encoder, "set_default_att_context_size"):
            asr_model.encoder.set_default_att_context_size(att_context_size=self.att_context_size)

        # chunk_size is set automatically for models trained for streaming.
        # For models trained for offline mode with full context, we need to pass the chunk_size explicitly.
        if self.chunk_size > 0:
            if self.shift_size < 0:
                shift_size = self.chunk_size
            else:
                shift_size = self.shift_size
            asr_model.encoder.setup_streaming_params(
                chunk_size=self.chunk_size, left_chunks=self.left_chunks, shift_size=shift_size
            )

        asr_model.eval()
        return asr_model

    def _get_best_hypothesis(self, encoded, encoded_len, partial_hypotheses=None):
        if self.decoder_type == "ctc":
            best_hyp = self.asr_model.decoding.ctc_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=True,
            )
        elif self.decoder_type == "rnnt":
            best_hyp = self.asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoded, encoded_len, return_hypotheses=True, partial_hypotheses=partial_hypotheses
            )
        else:
            raise ValueError("Decoder type not supported for this model.")
        return best_hyp

    def _get_tokens_and_probs_from_alignments(self, alignments):
        tokens = []
        probs = []

        # Debug: Check alignments structure
        if alignments is None:
            logger.debug(f"[ASR] Alignments: None")
            return tokens, probs

        logger.debug(f"[ASR] Alignments: type={type(alignments).__name__}, "
                     f"len={len(alignments) if hasattr(alignments, '__len__') else 'N/A'}")

        if self.decoder_type == "ctc":
            all_logits = alignments[0]
            all_tokens = alignments[1]
            logger.debug(f"[ASR] CTC alignments: logits_len={len(all_logits)}, tokens_len={len(all_tokens)}")
            blank_count = 0
            for i in range(len(all_tokens)):
                token_id = int(all_tokens[i])
                if token_id != self.blank_id:
                    tokens.append(token_id)
                    logits = all_logits[i]  # shape (vocab_size,)
                    probs_i = torch.softmax(logits, dim=-1)[token_id].item()
                    probs.append(probs_i)
                else:
                    blank_count += 1
            logger.debug(f"[ASR] CTC result: {len(tokens)} non-blank tokens, {blank_count} blank tokens")
        elif self.decoder_type == "rnnt":
            total_entries = 0
            blank_count = 0
            for t in range(len(alignments)):
                for u in range(len(alignments[t])):
                    total_entries += 1
                    logits, token_id = alignments[t][u]  # (logits, token_id)
                    token_id = int(token_id)
                    if token_id != self.blank_id:
                        tokens.append(token_id)
                        probs_i = torch.softmax(logits, dim=-1)[token_id].item()
                        probs.append(probs_i)
                    else:
                        blank_count += 1
            logger.debug(f"[ASR] RNNT result: {len(tokens)} non-blank tokens, {blank_count} blank tokens, "
                         f"total_entries={total_entries}")
        else:
            raise ValueError("Decoder type not supported for this model.")

        return tokens, probs

    def transcribe(self, audio: bytes, stream_id: str = "default") -> ASRResult:
        start_time = time.time()

        # =====================================================================
        # Session idle timeout check (safety net for stale tokens)
        # =====================================================================
        # If too much time has passed since the last transcribe call, this is
        # likely a new session. Clear all accumulated state to prevent stale
        # tokens from previous sessions contaminating the new session.
        time_since_last = start_time - self._last_transcribe_time
        if time_since_last > self._session_idle_timeout_secs and self._accumulated_tokens:
            logger.info(f"[ASR] Session idle timeout ({time_since_last:.1f}s > {self._session_idle_timeout_secs}s), "
                       f"clearing {len(self._accumulated_tokens)} stale tokens")
            self.reset_full()

        self._last_transcribe_time = start_time
        self._transcribe_call_count += 1
        call_num = self._transcribe_call_count
        logger.debug(f"[ASR] === Transcribe call #{call_num} ===")

        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Debug: Check audio input
        logger.debug(f"[ASR] Audio input: {len(audio)} bytes -> {len(audio_array)} samples, "
                     f"range=[{audio_array.min():.4f}, {audio_array.max():.4f}], "
                     f"abs_mean={np.abs(audio_array).mean():.6f}")

        self._audio_buffer.update(audio_array)

        # Debug: Check buffer state
        buffer_empty = self._audio_buffer.is_buffer_empty()
        logger.debug(f"[ASR] Buffer empty: {buffer_empty}")

        features = self._audio_buffer.get_feature_buffer()
        feature_lengths = torch.tensor([features.shape[1]], device=self.device)

        # Debug: Check feature buffer
        feature_min = features.min().item()
        feature_max = features.max().item()
        feature_mean = features.mean().item()
        non_zero_count = (features != -16.635).sum().item()
        logger.debug(f"[ASR] Features: shape={features.shape}, range=[{feature_min:.2f}, {feature_max:.2f}], "
                     f"mean={feature_mean:.2f}, non_zero_values={non_zero_count}")

        features = features.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = self.asr_model.encoder.cache_aware_stream_step(
                processed_signal=features,
                processed_signal_length=feature_lengths,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=False,
                drop_extra_pre_encoded=self.drop_extra_pre_encoded,
            )

        # Debug: Check encoded output
        encoded_min = encoded.min().item()
        encoded_max = encoded.max().item()
        encoded_mean = encoded.mean().item()
        encoded_std = encoded.std().item()
        logger.debug(f"[ASR] Encoded: shape={encoded.shape}, encoded_len={encoded_len.item()}, "
                     f"range=[{encoded_min:.4f}, {encoded_max:.4f}], mean={encoded_mean:.4f}, std={encoded_std:.4f}")

        # Debug: Check previous hypothesis state before decoding
        prev_hyp = self._previous_hypotheses[0] if self._previous_hypotheses else None
        if prev_hyp is not None:
            prev_y_len = len(prev_hyp.y_sequence) if hasattr(prev_hyp.y_sequence, '__len__') else 0
            prev_has_dec_state = prev_hyp.dec_state is not None
            prev_dec_state_info = "None"
            if prev_has_dec_state:
                if isinstance(prev_hyp.dec_state, (list, tuple)):
                    prev_dec_state_info = f"list/tuple[{len(prev_hyp.dec_state)}]"
                elif hasattr(prev_hyp.dec_state, 'shape'):
                    prev_dec_state_info = f"tensor{prev_hyp.dec_state.shape}"
            prev_last_token = prev_hyp.last_token
            logger.debug(f"[ASR] Previous hypothesis: y_sequence_len={prev_y_len}, "
                        f"dec_state={prev_dec_state_info}, last_token={prev_last_token}")
        else:
            logger.debug(f"[ASR] Previous hypothesis: None")

        best_hyp = self._get_best_hypothesis(encoded, encoded_len, partial_hypotheses=self._previous_hypotheses)

        # Debug: Check hypothesis (tensor-safe checks)
        if best_hyp is not None and len(best_hyp) > 0:
            hyp = best_hyp[0]
            # Safe check for alignments (could be tensor or list)
            if hyp.alignments is not None:
                alignments_len = len(hyp.alignments) if hasattr(hyp.alignments, '__len__') else 0
            else:
                alignments_len = 0
            # Safe check for y_sequence (could be tensor or list)
            if hyp.y_sequence is not None:
                if isinstance(hyp.y_sequence, torch.Tensor):
                    y_seq_len = hyp.y_sequence.numel()
                elif hasattr(hyp.y_sequence, '__len__'):
                    y_seq_len = len(hyp.y_sequence)
                else:
                    y_seq_len = 0
            else:
                y_seq_len = 0
            # Check dec_state (crucial for streaming)
            has_dec_state = hyp.dec_state is not None
            dec_state_info = "None"
            if has_dec_state:
                if isinstance(hyp.dec_state, (list, tuple)):
                    dec_state_info = f"list/tuple[{len(hyp.dec_state)}]"
                    if len(hyp.dec_state) > 0 and hasattr(hyp.dec_state[0], 'shape'):
                        dec_state_info += f", shape={hyp.dec_state[0].shape}"
                elif hasattr(hyp.dec_state, 'shape'):
                    dec_state_info = f"tensor{hyp.dec_state.shape}"
            last_token = hyp.last_token
            logger.debug(f"[ASR] Hypothesis: y_sequence_len={y_seq_len}, alignments_len={alignments_len}, "
                         f"score={hyp.score:.4f}, dec_state={dec_state_info}, last_token={last_token}")
        else:
            logger.debug(f"[ASR] Hypothesis: None or empty")

        self._previous_hypotheses = best_hyp
        self._cache_last_channel = cache_last_channel
        self._cache_last_time = cache_last_time
        self._cache_last_channel_len = cache_last_channel_len

        # Get current chunk tokens/probs from alignments
        hyp = best_hyp[0]
        chunk_tokens, probs = self._get_tokens_and_probs_from_alignments(hyp.alignments)

        # For Stateless decoders, we need to manually accumulate tokens across chunks
        # because the decoder resets y_sequence when dec_state is None
        if self._is_stateless_decoder:
            # Add current chunk tokens to accumulated tokens
            if chunk_tokens:
                self._accumulated_tokens.extend(chunk_tokens)
                logger.debug(f"[ASR] Stateless: Added {len(chunk_tokens)} tokens to accumulated "
                            f"(total={len(self._accumulated_tokens)})")
            tokens = self._accumulated_tokens
        else:
            # For Stateful decoders, use y_sequence (accumulated by decoder)
            if hyp.y_sequence is not None:
                if isinstance(hyp.y_sequence, torch.Tensor):
                    tokens = hyp.y_sequence.cpu().tolist()
                else:
                    tokens = list(hyp.y_sequence)
            else:
                tokens = []

        # Debug: Check tokens
        logger.debug(f"[ASR] Accumulated tokens: count={len(tokens)}, tokens={tokens[:10] if tokens else []}")
        logger.debug(f"[ASR] Chunk tokens from alignments: count={len(chunk_tokens)}, tokens={chunk_tokens[:10] if chunk_tokens else []}")

        text = self.get_text_from_tokens(tokens)

        # Debug: Final result
        logger.debug(f"[ASR] Final text: '{text}'")

        is_final = False
        eou_latency = None
        eob_latency = None
        eou_prob = None
        eob_prob = None
        current_timestamp = time.time()
        if self.eou_string in text or self.eob_string in text:
            is_final = True
            if self.eou_string in text:
                eou_latency = (
                    current_timestamp - self._last_transcript_timestamp if text.strip() == self.eou_string else 0.0
                )
                # Use chunk_tokens and probs (from alignments) for EOU probability
                # as they match 1:1 and EOU is detected in current chunk
                eou_prob = self.get_eou_probability(chunk_tokens, probs, self.eou_string)
            if self.eob_string in text:
                eob_latency = (
                    current_timestamp - self._last_transcript_timestamp if text.strip() == self.eob_string else 0.0
                )
                # Use chunk_tokens and probs for EOB probability
                eob_prob = self.get_eou_probability(chunk_tokens, probs, self.eob_string)
            self.reset_state(stream_id=stream_id)
        if text.strip():
            self._last_transcript_timestamp = current_timestamp

        processing_time = time.time() - start_time
        return ASRResult(
            text=text,
            is_final=is_final,
            eou_latency=eou_latency,
            eob_latency=eob_latency,
            eou_prob=eou_prob,
            eob_prob=eob_prob,
            processing_time=processing_time,
        )

    def reset_tokens_only(self, stream_id: str = "default"):
        """Soft reset: Reset tokens only, preserve encoder cache.

        Use this for STT-only mode when punctuation-based sentence boundary is detected.
        Preserves encoder cache to maintain recognition accuracy for the next chunk.

        Args:
            stream_id: Stream identifier (for future multi-stream support)
        """
        logger.debug(f"[ASR] reset_tokens_only called (soft reset)")
        logger.debug(f"[ASR] Before soft reset: accumulated_tokens={len(self._accumulated_tokens)}")

        # Reset token-related state only
        self._accumulated_tokens = []
        self._previous_hypotheses = self._get_blank_hypothesis()

        # Preserve encoder cache and audio buffer for continuous streaming
        # - self._reset_cache() NOT called - encoder cache preserved
        # - self._audio_buffer.reset() NOT called - buffer preserved

        logger.debug("[ASR] Soft reset complete: tokens cleared, encoder cache preserved")

    def reset_full(self, stream_id: str = "default"):
        """Hard reset: Reset all state including encoder cache.

        Use this when VAD detects silence (end of utterance).
        Resets all state for a fresh start on new utterance.

        Args:
            stream_id: Stream identifier (for future multi-stream support)
        """
        logger.debug(f"[ASR] reset_full called after {self._transcribe_call_count} transcribe calls")
        logger.debug(f"[ASR] Before full reset: accumulated_tokens={len(self._accumulated_tokens)}")
        self._audio_buffer.reset()
        self._reset_cache()  # Reset encoder cache
        self._previous_hypotheses = self._get_blank_hypothesis()
        self._accumulated_tokens = []  # Reset manual token accumulation for Stateless decoders
        self._last_transcript_timestamp = time.time()
        self._transcribe_call_count = 0
        logger.debug("[ASR] Full reset complete: all state cleared")

    def reset_state(self, stream_id: str = "default"):
        """Backward compatible alias for reset_full().

        Deprecated: Use reset_full() for hard reset or reset_tokens_only() for soft reset.
        """
        self.reset_full(stream_id)

    def get_eou_probability(self, tokens: List[int], probs: List[float], eou_string: str = "<EOU>") -> float:
        text_tokens = self.tokenizer.ids_to_tokens(tokens)
        eou_index = text_tokens.index(eou_string)
        return probs[eou_index]
