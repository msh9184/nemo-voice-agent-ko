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

import os
from typing import Any, Dict, Optional

from loguru import logger
from omegaconf import OmegaConf
from pipecat.audio.vad.silero import VADParams

from nemo.agents.voice_agent.pipecat.services.nemo.diar import NeMoDiarInputParams
from nemo.agents.voice_agent.pipecat.services.nemo.stt import NeMoSTTInputParams


class ConfigManager:
    """
    Manages configuration for the voice agent server.
    Handles loading, merging, and providing access to all configuration parameters.
    """

    def __init__(self, server_base_path: str, server_config_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the main server configuration file.
                        If None, uses default path from environment variable.
        """
        if not os.path.exists(server_base_path):
            raise FileNotFoundError(f"Server base path not found at {server_base_path}")

        self._server_base_path = server_base_path
        if server_config_path is not None:
            self._server_config_path = server_config_path
        else:
            self._server_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/default.yaml"

        if not os.path.exists(self._server_config_path):
            raise FileNotFoundError(f"Server configuration file not found at {self._server_config_path}")

        # Load model registry
        self.model_registry_path = f"{os.path.abspath(self._server_base_path)}/model_registry.yaml"
        self.model_registry = self._load_model_registry()

        # Load and process main configuration
        self.server_config = self._load_server_config()

        # Initialize configuration parameters
        self._initialize_config_parameters()

        self._generic_hf_llm_model_id = "hf_llm_generic"

        logger.info(f"Configuration loaded from: {self._server_config_path}")
        logger.info(f"Model registry loaded from: {self.model_registry_path}")

    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry from YAML file."""
        try:
            return OmegaConf.load(self.model_registry_path)
        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            raise ValueError(f"Failed to load model registry: {e}")

    def _load_server_config(self) -> OmegaConf:
        """Load and process the main server configuration."""
        server_config = OmegaConf.load(self._server_config_path)
        server_config = OmegaConf.to_container(server_config, resolve=True)
        server_config = OmegaConf.create(server_config)
        return server_config

    def _determine_service_mode(self) -> str:
        """Determine service mode based on enabled components.

        Returns:
            Service mode string identifying the active configuration.

        Service Modes:
            Voice Input Modes (STT enabled):
                - "full": STT + LLM + TTS (complete voice agent)
                - "stt_llm": STT + LLM (voice input, text output)
                - "stt_only": STT only (pure speech-to-text)

            Text Input Modes (STT disabled):
                - "llm_tts": LLM + TTS (text input, voice output)
                - "llm_only": LLM only (pure text chatbot)
                - "tts_only": TTS only (text-to-speech service)

            Error Mode:
                - "none": No services enabled (invalid configuration)
        """
        if self.STT_ENABLED:
            # Voice input modes
            if self.LLM_ENABLED and self.TTS_ENABLED:
                logger.info("Service Mode: FULL (STT + LLM + TTS)")
                return "full"
            elif self.LLM_ENABLED:
                logger.info("Service Mode: STT+LLM (Voice input, Text output)")
                return "stt_llm"
            else:
                logger.info("Service Mode: STT-ONLY (Pure speech-to-text)")
                return "stt_only"
        else:
            # Text input modes
            if self.LLM_ENABLED and self.TTS_ENABLED:
                logger.info("Service Mode: LLM+TTS (Text input, Voice output)")
                return "llm_tts"
            elif self.LLM_ENABLED:
                logger.info("Service Mode: LLM-ONLY (Pure text chatbot)")
                return "llm_only"
            elif self.TTS_ENABLED:
                logger.info("Service Mode: TTS-ONLY (Text-to-speech service)")
                return "tts_only"
            else:
                logger.warning("Service Mode: NONE - No services enabled!")
                return "none"

    def _initialize_config_parameters(self):
        """Initialize all configuration parameters from the loaded config."""
        # =====================================================================
        # Service Mode Configuration
        # =====================================================================
        # STT, LLM, and TTS can be independently enabled/disabled to support:
        #
        # Voice Input Modes (STT enabled):
        # - Full mode (STT+LLM+TTS): Complete voice agent (default)
        # - STT+LLM mode (STT+LLM): Voice input, text output chatbot
        # - STT-Only mode (STT): Pure speech-to-text
        #
        # Text Input Modes (STT disabled):
        # - LLM+TTS mode (LLM+TTS): Text input, voice output chatbot
        # - LLM-Only mode (LLM): Pure text chatbot
        # - TTS-Only mode (TTS): Text-to-speech service
        #
        self.STT_ENABLED = self.server_config.stt.get("enabled", True)
        self.LLM_ENABLED = self.server_config.llm.get("enabled", True)
        self.TTS_ENABLED = self.server_config.tts.get("enabled", True)

        # Determine service mode based on enabled components
        self._service_mode = self._determine_service_mode()
        self.SERVICE_MODE = self._service_mode

        # =====================================================================
        # Sample Rate Configuration
        # =====================================================================
        # INPUT_SAMPLE_RATE: Fixed at 16000Hz for STT/VAD/Diarization models
        # These models are trained on 16kHz audio and require this rate.
        self.INPUT_SAMPLE_RATE = 16000

        # OUTPUT_SAMPLE_RATE: Configurable for TTS output quality
        # Options:
        #   - "native": Use TTS model's native rate (best quality)
        #   - Specific value: 16000, 24000, 44100, etc.
        # Higher rates = better audio quality but more bandwidth
        # Lower rates = smaller frames, potentially lower latency
        audio_config = self.server_config.get("audio", {})
        output_rate_config = audio_config.get("output_sample_rate", "native")

        if output_rate_config == "native":
            # Will be set by TTS service based on model
            self.OUTPUT_SAMPLE_RATE = None  # Indicates "use TTS native rate"
        else:
            self.OUTPUT_SAMPLE_RATE = int(output_rate_config)

        # AUDIO_QUALITY_MODE: Convenience setting
        # "high_quality": Preserve TTS native rate (44100Hz for Fish Speech)
        # "low_latency": Resample to 16000Hz for smaller frames
        self.AUDIO_QUALITY_MODE = audio_config.get("quality_mode", "high_quality")

        # Legacy compatibility: SAMPLE_RATE used by STT/VAD
        self.SAMPLE_RATE = self.INPUT_SAMPLE_RATE

        # TTS-specific sample rate (set by TTS config or auto-detected)
        tts_sample_rate = self.server_config.get("tts", {}).get("sample_rate", None)
        self.TTS_SAMPLE_RATE = tts_sample_rate  # Can be None (auto-detect)

        self.RAW_AUDIO_FRAME_LEN_IN_SECS = 0.016
        self.SYSTEM_PROMPT = " ".join(
            [
                "You are a helpful AI agent named Lisa.",
                "Begin by warmly greeting the user and introducing yourself in one sentence.",
                "Keep your answers concise and to the point.",
            ]
        )

        # Transport configuration
        self.TRANSPORT_AUDIO_OUT_10MS_CHUNKS = self.server_config.transport.audio_out_10ms_chunks

        # VAD configuration (only needed when STT is enabled for voice input)
        if self.STT_ENABLED:
            self.vad_params = VADParams(
                confidence=self.server_config.vad.confidence,
                start_secs=self.server_config.vad.start_secs,
                stop_secs=self.server_config.vad.stop_secs,
                min_volume=self.server_config.vad.min_volume,
            )
        else:
            # Provide default VAD params even when STT is disabled
            # (may be needed for TTS-only mode audio level detection)
            self.vad_params = VADParams()

        # STT configuration (skip if disabled, but set defaults)
        if self.STT_ENABLED:
            self._configure_stt()
        else:
            self._set_stt_defaults()
            logger.info("STT disabled - using default STT configuration values")

        # Diarization configuration (only meaningful when STT is enabled)
        if self.STT_ENABLED:
            self._configure_diarization()
        else:
            self._set_diar_defaults()

        # Turn taking configuration (only meaningful when STT is enabled)
        if self.STT_ENABLED:
            self._configure_turn_taking()
        else:
            self._set_turn_taking_defaults()

        # LLM configuration (skip if disabled, but set defaults)
        if self.LLM_ENABLED:
            self._configure_llm()
        else:
            self._set_llm_defaults()
            logger.info("LLM disabled - using default LLM configuration values")

        # TTS configuration (skip if disabled, but set defaults)
        if self.TTS_ENABLED:
            self._configure_tts()
        else:
            self._set_tts_defaults()
            logger.info("TTS disabled - using default TTS configuration values")

    def _configure_stt(self):
        """Configure STT parameters."""
        self.STT_MODEL_PATH = self.server_config.stt.model
        self.STT_DEVICE = self.server_config.stt.device
        # Apply STT-specific configuration based on model type
        # Try to get STT config file name from server config first
        if self.server_config.stt.get("model_config", None) is not None:
            yaml_file_name = os.path.basename(self.server_config.stt.model_config)
        else:
            # Get STT configuration from registry
            if self.server_config.stt.type == "nemo" and "stt_en_fastconformer" in self.model_registry.stt_models:
                yaml_file_name = self.model_registry.stt_models[self.server_config.stt.model].yaml_id
            else:
                error_msg = f"STT model {self.STT_MODEL_PATH} with type {self.server_config.stt.type} is not supported configuration."
                logger.error(error_msg)
                raise ValueError(error_msg)

        stt_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/stt_configs/{yaml_file_name}"
        if not os.path.exists(stt_config_path):
            raise FileNotFoundError(f"STT config file not found at {stt_config_path}")
        stt_config = OmegaConf.load(stt_config_path)

        # merge stt config with server config
        for key in stt_config:
            if key in self.server_config.stt and self.server_config.stt[key] != stt_config[key]:
                logger.info(
                    f"STT config field `{key}` is overridden from `{self.server_config.stt[key]}` to `{stt_config[key]}` by {stt_config_path}"
                )
            self.server_config.stt[key] = stt_config[key]

        logger.info(f"Final STT config: {self.server_config.stt}")

        self.stt_params = NeMoSTTInputParams(
            att_context_size=self.server_config.stt.att_context_size,
            frame_len_in_secs=self.server_config.stt.frame_len_in_secs,
            raw_audio_frame_len_in_secs=self.RAW_AUDIO_FRAME_LEN_IN_SECS,
        )

        # Token filtering configuration for display output
        self.STT_FILTER_DISPLAY_TOKENS = self.server_config.stt.get("filter_display_tokens", None)
        if self.STT_FILTER_DISPLAY_TOKENS is not None:
            self.STT_FILTER_DISPLAY_TOKENS = OmegaConf.to_container(self.STT_FILTER_DISPLAY_TOKENS)
            logger.info(f"STT token filtering enabled: {len(self.STT_FILTER_DISPLAY_TOKENS)} tokens")

        self.STT_PRESERVE_RAW_FOR_LLM = self.server_config.stt.get("preserve_raw_for_llm", True)
        self.STT_EXTRACT_LANGUAGE_INFO = self.server_config.stt.get("extract_language_info", True)

    def _configure_diarization(self):
        """
        Configure diarization parameters.
        Currently only NeMo End-to-End Diarization is supported.
        """
        self.DIAR_MODEL = self.server_config.diar.model
        self.USE_DIAR = self.server_config.diar.enabled
        self.diar_params = NeMoDiarInputParams(
            frame_len_in_secs=self.server_config.diar.frame_len_in_secs,
            threshold=self.server_config.diar.threshold,
        )

    def _configure_turn_taking(self):
        """Configure turn taking parameters."""
        self.TURN_TAKING_BACKCHANNEL_PHRASES_PATH = self.server_config.turn_taking.backchannel_phrases_path
        self.TURN_TAKING_MAX_BUFFER_SIZE = self.server_config.turn_taking.max_buffer_size
        self.TURN_TAKING_BOT_STOP_DELAY = self.server_config.turn_taking.bot_stop_delay

    def _configure_llm(self):
        """Configure LLM parameters."""
        llm_model_id = self.server_config.llm.model

        # Try to get LLM config file name from server config first
        if self.server_config.llm.get("model_config", None) is not None:
            yaml_file_name = os.path.basename(self.server_config.llm.model_config)
        else:
            # Get LLM configuration from registry
            if llm_model_id in self.model_registry.llm_models:
                yaml_file_name = self.model_registry.llm_models[llm_model_id].yaml_id
            else:
                logger.warning(
                    f"LLM model {llm_model_id} is not included in the model registry. Using a generic HuggingFace LLM config."
                )
                yaml_file_name = self.model_registry.llm_models[self._generic_hf_llm_model_id].yaml_id

        # Load and merge LLM configuration
        llm_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/llm_configs/{yaml_file_name}"

        # Check reasoning support only if model is in registry
        if llm_model_id in self.model_registry.llm_models:
            if self.model_registry.llm_models[llm_model_id].get(
                "reasoning_supported", False
            ) and self.server_config.llm.get("enable_reasoning", False):
                llm_config_path = llm_config_path.replace(".yaml", "_think.yaml")

        if not os.path.exists(llm_config_path):
            raise FileNotFoundError(f"LLM config file not found at {llm_config_path}")
        logger.info(f"Loading LLM config from: {llm_config_path}")

        llm_config = OmegaConf.load(llm_config_path)
        # merge llm config with server config
        # print the override keys
        for key in llm_config:
            if key in self.server_config.llm and self.server_config.llm[key] != llm_config[key]:
                logger.info(
                    f"LLM config field `{key}` is overridden from `{self.server_config.llm[key]}` to `{llm_config[key]}` by {llm_config_path}"
                )
            self.server_config.llm[key] = llm_config[key]

        logger.info(f"Final LLM config: {self.server_config.llm}")

        # Configure system prompt
        self.SYSTEM_ROLE = self.server_config.llm.get("system_role", "system")
        if self.server_config.llm.get("system_prompt", None) is not None:
            system_prompt = self.server_config.llm.system_prompt
            if os.path.isfile(system_prompt):
                with open(system_prompt, "r") as f:
                    system_prompt = f.read()
            self.SYSTEM_PROMPT = system_prompt
        else:
            logger.info(f"No system prompt provided, using default system prompt: {self.SYSTEM_PROMPT}")

        if self.server_config.llm.get("system_prompt_suffix", None) is not None:
            self.SYSTEM_PROMPT += self.server_config.llm.system_prompt_suffix
            logger.info(f"Adding system prompt suffix: {self.server_config.llm.system_prompt_suffix}")

        logger.info(f"System prompt: {self.SYSTEM_PROMPT}")

    def _configure_tts(self):
        """Configure TTS parameters."""
        tts_type = self.server_config.tts.type
        # model field is optional for API-based TTS (fish_speech_api)
        tts_model_id = self.server_config.tts.get("model", None)

        # Try to get TTS config file name from server config first
        if self.server_config.tts.get("model_config", None) is not None:
            yaml_file_name = os.path.basename(self.server_config.tts.model_config)
            tts_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/tts_configs/{yaml_file_name}"
            if not os.path.exists(tts_config_path):
                raise FileNotFoundError(f"TTS config file not found at {tts_config_path}")
            tts_config = OmegaConf.load(tts_config_path)

            # merge tts config with server config
            for key in tts_config:
                if key in self.server_config.tts and self.server_config.tts[key] != tts_config[key]:
                    logger.info(
                        f"TTS config field `{key}` is overridden from `{self.server_config.tts[key]}` to `{tts_config[key]}` by {tts_config_path}"
                    )
                self.server_config.tts[key] = tts_config[key]
        elif tts_type in ["melo_korean", "kokoro_korean", "fish_speech", "fish_speech_api"]:
            # Korean TTS types and Fish Speech - use inline config from server config
            logger.info(f"Using TTS type: {tts_type}")
        elif tts_type == "kokoro":
            # Kokoro TTS - check if config exists in registry or use inline
            if tts_model_id in self.model_registry.get("tts_models", {}):
                yaml_file_name = self.model_registry.tts_models[tts_model_id].yaml_id
                tts_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/tts_configs/{yaml_file_name}"
                if os.path.exists(tts_config_path):
                    tts_config = OmegaConf.load(tts_config_path)
                    for key in tts_config:
                        if key in self.server_config.tts and self.server_config.tts[key] != tts_config[key]:
                            logger.info(
                                f"TTS config field `{key}` is overridden from `{self.server_config.tts[key]}` to `{tts_config[key]}` by {tts_config_path}"
                            )
                        self.server_config.tts[key] = tts_config[key]
            else:
                logger.info(f"Kokoro TTS model {tts_model_id} not in registry, using inline config")
        elif tts_type == "nemo" and "fastpitch-hifigan" in str(tts_model_id):
            # NeMo FastPitch-HiFiGAN TTS
            if tts_model_id in self.model_registry.get("tts_models", {}):
                yaml_file_name = self.model_registry.tts_models[tts_model_id].yaml_id
                tts_config_path = f"{os.path.abspath(self._server_base_path)}/server_configs/tts_configs/{yaml_file_name}"
                if not os.path.exists(tts_config_path):
                    raise FileNotFoundError(f"TTS config file not found at {tts_config_path}")
                tts_config = OmegaConf.load(tts_config_path)

                for key in tts_config:
                    if key in self.server_config.tts and self.server_config.tts[key] != tts_config[key]:
                        logger.info(
                            f"TTS config field `{key}` is overridden from `{self.server_config.tts[key]}` to `{tts_config[key]}` by {tts_config_path}"
                        )
                    self.server_config.tts[key] = tts_config[key]
            else:
                error_msg = f"TTS model {tts_model_id} not found in model registry"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"TTS model {tts_model_id} with type {tts_type} is not supported. Supported types: nemo, kokoro, melo_korean, kokoro_korean, fish_speech, fish_speech_api"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Final TTS config: {self.server_config.tts}")

        # Extract TTS parameters
        self.TTS_MAIN_MODEL_ID = self.server_config.tts.get("main_model_id", None)
        self.TTS_SUB_MODEL_ID = self.server_config.tts.get("sub_model_id", None)
        self.TTS_DEVICE = self.server_config.tts.get("device", "cpu")

        # Handle optional TTS parameters
        self.TTS_THINK_TOKENS = self.server_config.tts.get("think_tokens", None)
        if self.TTS_THINK_TOKENS is not None:
            self.TTS_THINK_TOKENS = OmegaConf.to_container(self.TTS_THINK_TOKENS)

        self.TTS_EXTRA_SEPARATOR = self.server_config.tts.get("extra_separator", None)
        if self.TTS_EXTRA_SEPARATOR is not None:
            self.TTS_EXTRA_SEPARATOR = OmegaConf.to_container(self.TTS_EXTRA_SEPARATOR)

    def _set_stt_defaults(self):
        """Set default STT attributes when STT is disabled.

        This ensures that code accessing STT-related attributes won't fail
        even when STT is not enabled (text input modes).
        """
        self.STT_MODEL_PATH = None
        self.STT_DEVICE = "cpu"
        self.stt_params = NeMoSTTInputParams(
            att_context_size=[70, 1],
            frame_len_in_secs=0.08,
            raw_audio_frame_len_in_secs=self.RAW_AUDIO_FRAME_LEN_IN_SECS,
        )
        self.STT_FILTER_DISPLAY_TOKENS = None
        self.STT_PRESERVE_RAW_FOR_LLM = True
        self.STT_EXTRACT_LANGUAGE_INFO = True

    def _set_diar_defaults(self):
        """Set default diarization attributes when STT is disabled.

        Diarization is only meaningful with voice input, so we set
        safe defaults when STT is disabled.
        """
        self.DIAR_MODEL = None
        self.USE_DIAR = False
        self.diar_params = NeMoDiarInputParams(
            frame_len_in_secs=0.08,
            threshold=0.5,
        )

    def _set_turn_taking_defaults(self):
        """Set default turn-taking attributes when STT is disabled.

        Turn-taking manages conversation flow for voice interactions.
        Text input modes don't need turn-taking (Enter key = end of turn).
        """
        self.TURN_TAKING_BACKCHANNEL_PHRASES_PATH = None
        self.TURN_TAKING_MAX_BUFFER_SIZE = 10
        self.TURN_TAKING_BOT_STOP_DELAY = 0.1

    def _set_llm_defaults(self):
        """Set default LLM attributes when LLM is disabled.

        This ensures that code accessing LLM-related attributes won't fail
        even when LLM is not enabled.
        """
        self.SYSTEM_ROLE = "system"
        # SYSTEM_PROMPT is already set in _initialize_config_parameters()

    def _set_tts_defaults(self):
        """Set default TTS attributes when TTS is disabled.

        This ensures that code accessing TTS-related attributes won't fail
        even when TTS is not enabled.
        """
        self.TTS_MAIN_MODEL_ID = None
        self.TTS_SUB_MODEL_ID = None
        self.TTS_DEVICE = "cpu"
        self.TTS_THINK_TOKENS = None
        self.TTS_EXTRA_SEPARATOR = None

    def get_server_config(self) -> OmegaConf:
        """Get the complete server configuration."""
        return self.server_config

    def get_model_registry(self) -> Dict[str, Any]:
        """Get the model registry configuration."""
        return self.model_registry

    def get_vad_params(self) -> VADParams:
        """Get VAD parameters."""
        return self.vad_params

    def get_stt_params(self) -> NeMoSTTInputParams:
        """Get STT parameters."""
        return self.stt_params

    def get_diar_params(self) -> NeMoDiarInputParams:
        """Get diarization parameters."""
        return self.diar_params
