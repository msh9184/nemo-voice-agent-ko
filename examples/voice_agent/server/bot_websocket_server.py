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
import copy
import os
import signal
import sys

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIAction, RTVIConfig, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer

from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi import RTVIObserver
from nemo.agents.voice_agent.pipecat.services.nemo.diar import NemoDiarService
from nemo.agents.voice_agent.pipecat.services.nemo.llm import get_llm_service_from_config
from nemo.agents.voice_agent.pipecat.services.nemo.stt import NemoSTTService
from nemo.agents.voice_agent.pipecat.services.nemo.tts import (
    FishSpeechAPIService,
    FishSpeechKoreanService,
    KokoroTTSService,
    MeloTTSKoreanService,
    NeMoFastPitchHiFiGANTTSService,
)
from nemo.agents.voice_agent.pipecat.services.nemo.turn_taking import NeMoTurnTakingService
from nemo.agents.voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from nemo.agents.voice_agent.pipecat.utils.text.simple_text_aggregator import SimpleSegmentedTextAggregator
from nemo.agents.voice_agent.pipecat.processors.display_processor import VoiceAgentDisplayProcessor
from nemo.agents.voice_agent.pipecat.utils.realtime_display import VoiceAgentDisplay, get_voice_agent_display
from nemo.agents.voice_agent.utils.config_manager import ConfigManager
# STT-only mode aggregators (used when LLM/TTS are disabled)
from nemo.agents.voice_agent.pipecat.services.nemo.stt_aggregator import STTTextAggregator
from nemo.agents.voice_agent.pipecat.services.nemo.speaker_aware_aggregator import SpeakerAwareAggregator
from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi import SpeakerFrameProcessor


def _is_websocket_open(websocket) -> bool:
    """Check if a websocket connection is open (compatible with websockets v10+).

    The websockets library v10+ removed the `.closed` attribute.
    This helper function provides a compatible way to check connection status.

    Args:
        websocket: WebSocket connection object

    Returns:
        True if the websocket is open and can send messages, False otherwise
    """
    if websocket is None:
        return False

    try:
        # websockets v10+ uses State enum
        # Try to import and check against State.OPEN
        try:
            from websockets.protocol import State
            if hasattr(websocket, 'state'):
                return websocket.state == State.OPEN
        except ImportError:
            pass

        # websockets v10+ alternative: check state directly
        if hasattr(websocket, 'state'):
            # State enum: CONNECTING=0, OPEN=1, CLOSING=2, CLOSED=3
            state = websocket.state
            if hasattr(state, 'value'):
                return state.value == 1  # OPEN
            elif hasattr(state, 'name'):
                return state.name == 'OPEN'
            else:
                return str(state) == 'State.OPEN' or state == 1

        # Fallback for older websockets versions
        if hasattr(websocket, 'closed'):
            return not websocket.closed

        # Last resort: assume open if no definitive indicator
        return True

    except Exception as e:
        logger.debug(f"Error checking websocket status: {e}")
        return False


def setup_logging():
    # Configure loguru to output to both console and file
    # Use INFO level for console (cleaner output), DEBUG level for file
    logger.remove()  # Remove default handler

    # Filter function to suppress noisy pipecat warnings
    def console_filter(record):
        # Filter out DEBUG level
        if record["level"].name == "DEBUG":
            return False
        # Suppress StartInterruptionFrame warnings (noisy, expected behavior)
        message = record.get("message", "")
        if "StartInterruptionFrame" in message:
            return False
        # Suppress other noisy pipecat frame warnings
        if "Unhandled upstream frame" in message and "Frame" in message:
            return False
        return True

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",  # Changed from DEBUG to INFO for cleaner console output
        filter=console_filter,
    )

    logger.add("bot_server.log", rotation="1 day", level="DEBUG")  # Keep DEBUG for file


setup_logging()

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()

# Initialize configuration manager
config_manager = ConfigManager(
    server_base_path=os.path.dirname(__file__), server_config_path=os.environ.get("SERVER_CONFIG_PATH", None)
)
server_config = config_manager.get_server_config()

logger.info(f"Server config: {server_config}")

# Access configuration parameters from ConfigManager
# =============================================================================
# Service Mode Configuration (read early to enable conditional setup)
# =============================================================================
# STT, LLM, and TTS can be independently enabled/disabled to support:
#
# Voice Input Modes (STT enabled):
# - Full mode (STT+LLM+TTS): Complete voice agent (default)
# - STT+LLM mode (STT+LLM): Voice input, text output chatbot
# - STT-Only mode (STT): Pure speech-to-text service
#
# Text Input Modes (STT disabled):
# - LLM+TTS mode (LLM+TTS): Text input, voice output chatbot
# - LLM-Only mode (LLM): Pure text chatbot
# - TTS-Only mode (TTS): Text-to-speech service
#
STT_ENABLED = config_manager.STT_ENABLED
LLM_ENABLED = config_manager.LLM_ENABLED
TTS_ENABLED = config_manager.TTS_ENABLED
SERVICE_MODE = config_manager.SERVICE_MODE

logger.info(f"Service Mode Configuration:")
logger.info(f"  Service Mode: {SERVICE_MODE}")
logger.info(f"  STT Enabled: {STT_ENABLED}")
logger.info(f"  LLM Enabled: {LLM_ENABLED}")
logger.info(f"  TTS Enabled: {TTS_ENABLED}")

# =============================================================================
# Sample Rate Configuration
# =============================================================================
# INPUT_SAMPLE_RATE: For STT/VAD/Diarization (fixed at 16kHz for model compatibility)
# OUTPUT_SAMPLE_RATE: For TTS output (configurable: "native" or specific rate)
# SAMPLE_RATE: Legacy compatibility (same as INPUT_SAMPLE_RATE)
INPUT_SAMPLE_RATE = config_manager.INPUT_SAMPLE_RATE  # 16000 (fixed)
OUTPUT_SAMPLE_RATE = config_manager.OUTPUT_SAMPLE_RATE  # None="native", or specific rate
AUDIO_QUALITY_MODE = config_manager.AUDIO_QUALITY_MODE  # "high_quality" or "low_latency"
TTS_SAMPLE_RATE = config_manager.TTS_SAMPLE_RATE  # TTS-specific rate from config

# Legacy compatibility
SAMPLE_RATE = config_manager.SAMPLE_RATE  # Same as INPUT_SAMPLE_RATE

logger.info(f"Audio Configuration:")
logger.info(f"  Input Sample Rate (STT/VAD): {INPUT_SAMPLE_RATE}Hz")
if TTS_ENABLED:
    logger.info(f"  Output Sample Rate: {OUTPUT_SAMPLE_RATE if OUTPUT_SAMPLE_RATE else 'native (TTS auto)'}")
    logger.info(f"  Quality Mode: {AUDIO_QUALITY_MODE}")

# Determine TTS output sample rate early for WebSocket transport configuration
# This is needed because the transport must know the audio output sample rate
# before TTS service is initialized
TTS_TYPE = config_manager.server_config.tts.type if TTS_ENABLED else "none"
TTS_NATIVE_RATES = {
    "nemo": 22050,
    "kokoro": 24000,
    "kokoro_korean": 24000,
    "melo_korean": 44100,
    "fish_speech": 24000,
    "fish_speech_api": 44100,
    "cosyvoice": 22050,
    "cosyvoice3": 22050,
}
TTS_OUTPUT_SAMPLE_RATE = OUTPUT_SAMPLE_RATE if OUTPUT_SAMPLE_RATE else TTS_NATIVE_RATES.get(TTS_TYPE, 16000)
if TTS_ENABLED:
    logger.info(f"  TTS Output Sample Rate: {TTS_OUTPUT_SAMPLE_RATE}Hz (type: {TTS_TYPE})")

RAW_AUDIO_FRAME_LEN_IN_SECS = config_manager.RAW_AUDIO_FRAME_LEN_IN_SECS
SYSTEM_PROMPT = config_manager.SYSTEM_PROMPT
SYSTEM_ROLE = config_manager.SYSTEM_ROLE

# Transport configuration
TRANSPORT_AUDIO_OUT_10MS_CHUNKS = config_manager.TRANSPORT_AUDIO_OUT_10MS_CHUNKS

# VAD configuration (only used when STT is enabled)
vad_params = config_manager.get_vad_params()

# STT configuration (only used when STT is enabled)
if STT_ENABLED:
    STT_MODEL_PATH = config_manager.STT_MODEL_PATH
    STT_DEVICE = config_manager.STT_DEVICE
    stt_params = config_manager.get_stt_params()
else:
    STT_MODEL_PATH = None
    STT_DEVICE = None
    stt_params = None
    logger.info("STT service DISABLED - text input mode active")

# Diarization configuration (only meaningful when STT is enabled)
if STT_ENABLED:
    DIAR_MODEL = config_manager.DIAR_MODEL
    USE_DIAR = config_manager.USE_DIAR
    diar_params = config_manager.get_diar_params()
else:
    DIAR_MODEL = None
    USE_DIAR = False
    diar_params = None

# Turn taking configuration (only meaningful when STT is enabled)
if STT_ENABLED:
    TURN_TAKING_BACKCHANNEL_PHRASES_PATH = config_manager.TURN_TAKING_BACKCHANNEL_PHRASES_PATH
    TURN_TAKING_MAX_BUFFER_SIZE = config_manager.TURN_TAKING_MAX_BUFFER_SIZE
    TURN_TAKING_BOT_STOP_DELAY = config_manager.TURN_TAKING_BOT_STOP_DELAY
else:
    # Text input doesn't need turn-taking (Enter key = end of turn)
    TURN_TAKING_BACKCHANNEL_PHRASES_PATH = None
    TURN_TAKING_MAX_BUFFER_SIZE = 10
    TURN_TAKING_BOT_STOP_DELAY = 0.1

# TTS configuration (TTS_TYPE already defined earlier for sample rate calculation)
# Only access TTS model IDs if TTS is enabled
if TTS_ENABLED:
    TTS_MAIN_MODEL_ID = config_manager.TTS_MAIN_MODEL_ID
    TTS_SUB_MODEL_ID = config_manager.TTS_SUB_MODEL_ID
    TTS_DEVICE = config_manager.TTS_DEVICE
    TTS_THINK_TOKENS = config_manager.TTS_THINK_TOKENS
    TTS_EXTRA_SEPARATOR = config_manager.TTS_EXTRA_SEPARATOR
else:
    TTS_MAIN_MODEL_ID = None
    TTS_SUB_MODEL_ID = None
    TTS_DEVICE = None
    TTS_THINK_TOKENS = None
    TTS_EXTRA_SEPARATOR = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


async def run_bot_websocket_server(host: str = "0.0.0.0", port: int = 8765):
    logger.info(f"Starting websocket server on {host}:{port}")
    logger.info(f"Server configured to run indefinitely with no timeouts, use Ctrl+C to quit.")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Initializing WebSocket server transport...")
    logger.info("Server configured to run indefinitely with no timeouts")

    """
    NO-TIMEOUT CONFIGURATION:
    - session_timeout=None: Disables WebSocket session timeout
    - idle_timeout=None: Disables pipeline idle timeout  
    - asyncio.wait_for(timeout=None): No timeout on pipeline runner
    - Server will run indefinitely until manually stopped (Ctrl+C)
    """

    # ==========================================================================
    # VAD Analyzer Setup (only needed for voice input modes)
    # ==========================================================================
    vad_analyzer = None
    if STT_ENABLED:
        vad_analyzer = SileroVADAnalyzer(
            sample_rate=SAMPLE_RATE,
            params=vad_params,
        )
        logger.info("VAD analyzer initialized")
    else:
        logger.info("VAD analyzer SKIPPED - text input mode active")

    # ==========================================================================
    # WebSocket Transport Setup
    # ==========================================================================
    # Audio input is only needed when STT is enabled (voice input modes)
    # Audio output is only needed when TTS is enabled
    audio_in_enabled = STT_ENABLED
    audio_out_enabled = TTS_ENABLED
    logger.info(f"Audio input enabled: {audio_in_enabled}, Audio output enabled: {audio_out_enabled}")

    ws_transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=audio_in_enabled,
            audio_out_enabled=audio_out_enabled,
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            session_timeout=None,  # Disable session timeout
            audio_in_sample_rate=SAMPLE_RATE if STT_ENABLED else 16000,
            # Text input modes: always allow user frames immediately (no VAD gating)
            # Voice input modes: use existing logic
            can_create_user_frames=(not STT_ENABLED) or (not LLM_ENABLED) or (TURN_TAKING_BACKCHANNEL_PHRASES_PATH is None),
            audio_out_10ms_chunks=TRANSPORT_AUDIO_OUT_10MS_CHUNKS,
        ),
        host=host,
        port=port,
    )

    # ==========================================================================
    # STT Service Setup (only for voice input modes)
    # ==========================================================================
    stt = None
    if STT_ENABLED:
        logger.info("Initializing STT service...")

        # Turn-taking is only needed when LLM is enabled for conversation flow control
        # STT-only mode uses aggregators instead of turn-taking
        stt_has_turn_taking = LLM_ENABLED
        # Audio passthrough is needed for diarization, regardless of mode
        stt_audio_passthrough = USE_DIAR or LLM_ENABLED

        # STT-only mode stability: Enable soft reset for continuous streaming
        # When STT-only (no LLM), use punctuation+word-count based token reset
        # to prevent OOM from unbounded token accumulation
        is_stt_only_mode = STT_ENABLED and not LLM_ENABLED
        stt_min_words_for_soft_reset = config_manager.server_config.stt.get("min_words_for_soft_reset", 5)
        stt_max_words_for_soft_reset = config_manager.server_config.stt.get("max_words_for_soft_reset", 20)

        stt = NemoSTTService(
            model=STT_MODEL_PATH,
            device=STT_DEVICE,
            params=stt_params,
            sample_rate=SAMPLE_RATE,
            audio_passthrough=stt_audio_passthrough,
            has_turn_taking=stt_has_turn_taking,
            backend="legacy",
            decoder_type="rnnt",
            # Token filtering options from config
            filter_display_tokens=config_manager.STT_FILTER_DISPLAY_TOKENS,
            preserve_raw_for_llm=config_manager.STT_PRESERVE_RAW_FOR_LLM,
            extract_language_info=config_manager.STT_EXTRACT_LANGUAGE_INFO,
            # STT-only mode stability parameters
            stt_only_mode=is_stt_only_mode,
            min_words_for_soft_reset=stt_min_words_for_soft_reset,
            max_words_for_soft_reset=stt_max_words_for_soft_reset,
        )
        logger.info(f"STT service initialized (turn_taking={stt_has_turn_taking}, audio_passthrough={stt_audio_passthrough}, stt_only_mode={is_stt_only_mode})")
    else:
        logger.info("STT service SKIPPED - text input mode active")

    # ==========================================================================
    # Diarization Service Setup (only for voice input modes with diarization)
    # ==========================================================================
    diar = None
    if STT_ENABLED and USE_DIAR:
        diar = NemoDiarService(
            model=DIAR_MODEL,
            device=STT_DEVICE,
            params=diar_params,
            sample_rate=SAMPLE_RATE,
            backend="legacy",
            enabled=USE_DIAR,
        )
        logger.info("Diarization service initialized")
    elif USE_DIAR and not STT_ENABLED:
        logger.info("Diarization SKIPPED - not meaningful without STT")

    # =============================================================================
    # Conditional Service Initialization based on Service Mode
    # =============================================================================

    # Turn-taking is only used when both STT and LLM are enabled (voice conversation mode)
    # Text input modes don't need turn-taking (Enter key = end of turn)
    turn_taking = None
    if STT_ENABLED and LLM_ENABLED:
        turn_taking = NeMoTurnTakingService(
            use_vad=True,
            use_diar=USE_DIAR,
            max_buffer_size=TURN_TAKING_MAX_BUFFER_SIZE,
            bot_stop_delay=TURN_TAKING_BOT_STOP_DELAY,
            backchannel_phrases=TURN_TAKING_BACKCHANNEL_PHRASES_PATH,
        )
        logger.info("Turn taking service initialized")
    elif not STT_ENABLED and LLM_ENABLED:
        logger.info("Turn taking SKIPPED - text input mode (Enter key = end of turn)")

    # LLM service (only initialized when enabled)
    llm = None
    context = None
    context_aggregator = None
    user_context_aggregator = None
    assistant_context_aggregator = None
    original_messages = None
    original_context = None

    if LLM_ENABLED:
        logger.info("Initializing LLM service...")
        llm = get_llm_service_from_config(server_config.llm)
        logger.info("LLM service initialized")
    else:
        logger.info("LLM service DISABLED - skipping initialization")

    # Text aggregator for TTS (only used when TTS is enabled AND LLM is enabled)
    # In TTS-Only mode (no LLM), text comes as complete messages, not streaming tokens.
    # So we don't need text aggregation - it would incorrectly buffer partial sentences.
    # The aggregator is designed for LLM streaming output where tokens arrive one at a time.
    #
    # IMPORTANT: pipecat's TTSService has TWO aggregation mechanisms:
    # 1. text_aggregator: Custom aggregator passed via constructor
    # 2. aggregate_sentences: Boolean flag (default True) that creates internal aggregator
    # For TTS-Only mode, we must disable BOTH to ensure complete text passthrough.
    text_aggregator = None
    tts_aggregate_sentences = True  # Default: aggregate sentences for LLM streaming output

    if TTS_ENABLED and LLM_ENABLED:
        text_aggregator = SimpleSegmentedTextAggregator(punctuation_marks=TTS_EXTRA_SEPARATOR)
        tts_aggregate_sentences = True
        logger.info("Text aggregator enabled for LLM+TTS mode (streaming sentence segmentation)")
    elif TTS_ENABLED:
        # TTS-Only mode: Disable ALL sentence aggregation for complete text passthrough
        text_aggregator = None
        tts_aggregate_sentences = False
        logger.info("Text aggregator DISABLED for TTS-Only mode (aggregate_sentences=False, complete text passthrough)")

    # STT-only mode aggregators (used when LLM is disabled)
    stt_aggregator = None
    speaker_aggregator = None
    speaker_frame_processor = None

    # TTS service (only initialized when enabled)
    tts = None
    if TTS_ENABLED:
        if TTS_TYPE == "nemo":
            tts = NeMoFastPitchHiFiGANTTSService(
                fastpitch_model=TTS_MAIN_MODEL_ID,
                hifigan_model=TTS_SUB_MODEL_ID,
                device=TTS_DEVICE,
                text_aggregator=text_aggregator,
                aggregate_sentences=tts_aggregate_sentences,
                think_tokens=TTS_THINK_TOKENS,
            )
        elif TTS_TYPE == "kokoro":
            tts = KokoroTTSService(
                voice=TTS_SUB_MODEL_ID,
                device=TTS_DEVICE,
                speed=config_manager.server_config.tts.speed,
                text_aggregator=text_aggregator,
                aggregate_sentences=tts_aggregate_sentences,
                think_tokens=TTS_THINK_TOKENS,
            )
        elif TTS_TYPE == "melo_korean":
            tts = MeloTTSKoreanService(
                device=TTS_DEVICE,
                speed=config_manager.server_config.tts.get("speed", 1.0),
                text_aggregator=text_aggregator,
                aggregate_sentences=tts_aggregate_sentences,
                think_tokens=TTS_THINK_TOKENS,
            )
        elif TTS_TYPE == "kokoro_korean":
            # Kokoro with Korean language support (lang_code='k')
            tts = KokoroTTSService(
                lang_code="k",  # Korean language code
                voice=TTS_SUB_MODEL_ID if TTS_SUB_MODEL_ID else "kf_yujeong",  # Korean voice
                device=TTS_DEVICE,
                speed=config_manager.server_config.tts.get("speed", 1.0),
                text_aggregator=text_aggregator,
                aggregate_sentences=tts_aggregate_sentences,
                think_tokens=TTS_THINK_TOKENS,
            )
        elif TTS_TYPE == "fish_speech":
            # OpenAudio S1-mini with Korean language support (RLHF trained)
            tts = FishSpeechKoreanService(
                model_path=TTS_MAIN_MODEL_ID if TTS_MAIN_MODEL_ID else "/path/to/model",
                device=TTS_DEVICE,
                reference_audio_path=config_manager.server_config.tts.get("reference_audio_path", None),
                reference_audio_text=config_manager.server_config.tts.get("reference_audio_text", None),
                streaming=config_manager.server_config.tts.get("streaming", True),
                use_subprocess=config_manager.server_config.tts.get("use_subprocess", True),
                fish_speech_repo=config_manager.server_config.tts.get("fish_speech_repo", None),
                text_aggregator=text_aggregator,
                aggregate_sentences=tts_aggregate_sentences,
                think_tokens=TTS_THINK_TOKENS,
            )
        elif TTS_TYPE == "fish_speech_api":
            # OpenAudio S1-mini via HTTP API (RECOMMENDED for low latency)
            tts = FishSpeechAPIService(
                api_url=config_manager.server_config.tts.get("api_url", "http://localhost:8080"),
                device=TTS_DEVICE,
                sample_rate=TTS_OUTPUT_SAMPLE_RATE,
                reference_audio_path=config_manager.server_config.tts.get("reference_audio_path", None),
                reference_audio_text=config_manager.server_config.tts.get("reference_audio_text", None),
                reference_id=config_manager.server_config.tts.get("reference_id", None),
                use_memory_cache=config_manager.server_config.tts.get("use_memory_cache", "on"),
                timeout=config_manager.server_config.tts.get("timeout", 60),
                text_aggregator=text_aggregator,
                aggregate_sentences=tts_aggregate_sentences,
                think_tokens=TTS_THINK_TOKENS,
            )
        else:
            raise ValueError(f"Invalid TTS type: {TTS_TYPE}. Supported types: nemo, kokoro, melo_korean, kokoro_korean, fish_speech, fish_speech_api")
        logger.info("TTS service initialized")
    else:
        logger.info("TTS service DISABLED - skipping initialization")

    # Context and aggregators (only needed when LLM is enabled)
    if LLM_ENABLED and llm is not None:
        context = OpenAILLMContext(
            [
                {
                    "role": SYSTEM_ROLE,
                    "content": SYSTEM_PROMPT,
                }
            ],
        )

        original_messages = copy.deepcopy(context.get_messages())
        original_context = copy.deepcopy(context)
        original_context.set_llm_adapter(llm.get_llm_adapter())

        context_aggregator = llm.create_context_aggregator(context)
        user_context_aggregator = context_aggregator.user()
        assistant_context_aggregator = context_aggregator.assistant()
    else:
        # STT-only mode: Initialize aggregators for text output
        if USE_DIAR and diar is not None:
            # Speaker-aware aggregator for multi-speaker scenarios
            speaker_aggregator = SpeakerAwareAggregator(
                max_speakers=4,
                use_speaker_boundary=True,
                use_vad_boundary=True,
                use_punctuation_boundary=True,
                min_words_for_speaker_commit=1,
                min_words_for_punctuation_boundary=3,
                max_words_before_commit=20,
                sentence_end_punctuation=".?!。？！",
                silence_timeout_secs=3.0,
                silence_check_interval_secs=0.5,
                eou_string="<EOU>",
                eob_string="<EOB>",
                use_realtime_display=True,
            )
            logger.info("Speaker-Aware Aggregator initialized for STT-only mode")
        else:
            # Standard STT-only aggregator (no speaker tracking)
            stt_aggregator = STTTextAggregator(
                eou_string="<EOU>",
                eob_string="<EOB>",
                sentence_end_punctuation=".?!。？！",
                use_vad_boundary=True,
                use_punctuation_boundary=True,
                min_words_for_punctuation_boundary=3,
                max_words_before_commit=20,
                silence_timeout_secs=3.0,
                silence_check_interval_secs=0.5,
            )
            logger.info("STT Text Aggregator initialized for STT-only mode")

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Add reset action to RTVI processor
    async def reset_context_handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, any]) -> bool:
        """Reset conversation context (handles both full mode and STT-only mode)"""
        logger.info("Resetting conversation context...")
        try:
            # Full mode: Reset LLM context aggregators
            if LLM_ENABLED and user_context_aggregator is not None and assistant_context_aggregator is not None:
                user_context_aggregator.reset()
                assistant_context_aggregator.reset()
                user_context_aggregator.set_messages(copy.deepcopy(original_messages))
                assistant_context_aggregator.set_messages(copy.deepcopy(original_messages))
                if text_aggregator is not None:
                    await text_aggregator.reset()
            # STT-only mode: Reset aggregators
            else:
                if stt_aggregator is not None:
                    stt_aggregator.reset()
                if speaker_aggregator is not None:
                    speaker_aggregator.reset()
            # Reset diarization in all modes
            if diar is not None:
                diar.reset()
            logger.info("Conversation context reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting context: {e}")
            return False

    reset_action = RTVIAction(
        service="context",
        action="reset",
        result="bool",
        arguments=[],
        handler=reset_context_handler,
    )
    rtvi.register_action(reset_action)

    # ==========================================================================
    # Text Input RTVI Action (for text input modes: LLM-Only, TTS-Only, LLM+TTS)
    # ==========================================================================
    # This action allows clients to send text directly, bypassing STT.
    # It replaces the STT module for text input modes.
    async def text_input_handler(
        rtvi_processor: RTVIProcessor, service: str, arguments: list
    ) -> dict:
        """Handle text input from client (bypasses STT).

        This handler mimics what STT would output:
        - For LLM modes: Adds text to LLM context and triggers response
        - For TTS-only mode: Sends text directly to TTS for synthesis

        Args:
            rtvi_processor: RTVI processor instance
            service: Service name ("text_input")
            arguments: List of argument dicts [{"name": "text", "value": "..."}]

        Returns:
            Dict with success status and optional error message
        """
        from pipecat.frames.frames import TextFrame

        # Debug: Log raw arguments format
        logger.debug(f"[TEXT INPUT] Raw arguments type: {type(arguments)}, value: {arguments}")

        # Extract text from arguments (support multiple formats)
        text = ""

        # Format 1: arguments is a dict with "text" key (RTVI style)
        # e.g., {"text": "안녕하세요"}
        if isinstance(arguments, dict):
            text = arguments.get("text", "").strip()
            logger.debug(f"[TEXT INPUT] Format 1 (dict): extracted '{text}'")

        # Format 2: arguments is a list
        elif isinstance(arguments, list):
            for arg in arguments:
                # Format 2a: [{"name": "text", "value": "안녕"}]
                if isinstance(arg, dict):
                    if arg.get("name") == "text":
                        text = arg.get("value", "").strip()
                        logger.debug(f"[TEXT INPUT] Format 2a (named dict): extracted '{text}'")
                        break
                    # Format 2b: [{"text": "안녕"}] - dict with text key directly
                    elif "text" in arg:
                        text = arg.get("text", "").strip()
                        logger.debug(f"[TEXT INPUT] Format 2b (text key dict): extracted '{text}'")
                        break
                    # Format 2c: [{"value": "안녕"}] - value only
                    elif "value" in arg:
                        text = arg.get("value", "").strip()
                        logger.debug(f"[TEXT INPUT] Format 2c (value only dict): extracted '{text}'")
                        break
                # Format 2d: ["안녕"] - simple string in list
                elif isinstance(arg, str):
                    text = arg.strip()
                    logger.debug(f"[TEXT INPUT] Format 2d (string in list): extracted '{text}'")
                    break

        # Format 3: arguments is a single string
        elif isinstance(arguments, str):
            text = arguments.strip()
            logger.debug(f"[TEXT INPUT] Format 3 (string): extracted '{text}'")

        if not text:
            logger.warning(f"[TEXT INPUT] Empty text received. Arguments: {arguments}")
            return {"success": False, "error": "Empty text"}

        # Log received text (truncate long messages)
        display_text = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"[TEXT INPUT] Received: '{display_text}'")

        try:
            # Reset LLM accumulated state before processing new text input
            # This prevents response accumulation in LLM-Only mode
            # (where BotStoppedSpeakingFrame never fires to reset the state)
            await rtvi_observer.reset_llm_state()

            if LLM_ENABLED:
                # =============================================================
                # LLM-Only or LLM+TTS mode
                # =============================================================
                # Add user message to LLM context and trigger response generation
                if user_context_aggregator is not None and context is not None:
                    # Use the outer context variable directly (OpenAILLMContext)
                    context.add_message({"role": "user", "content": text})

                    # Trigger LLM processing
                    await task.queue_frames([
                        user_context_aggregator.get_context_frame()
                    ])

                    # Display user message in console
                    voice_display.user_text(text)

                    logger.info(f"[TEXT INPUT] Queued to LLM context")
                else:
                    logger.error("[TEXT INPUT] user_context_aggregator not available")
                    return {"success": False, "error": "LLM context not available"}

            elif TTS_ENABLED:
                # =============================================================
                # TTS-Only mode
                # =============================================================
                # Send text directly to TTS for synthesis (no LLM processing)
                await task.queue_frames([TextFrame(text=text)])

                # Display text in console
                voice_display.bot_text(f"[TTS] {text}")

                logger.info(f"[TEXT INPUT] Queued to TTS directly")

            else:
                # No processing available (should not happen with valid config)
                logger.error("[TEXT INPUT] Neither LLM nor TTS is enabled")
                return {"success": False, "error": "No output service enabled"}

            return {"success": True, "text": text}

        except Exception as e:
            logger.error(f"[TEXT INPUT] Error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # Register text_input action
    text_input_action = RTVIAction(
        service="text_input",
        action="send",
        result="object",
        arguments=[
            {"name": "text", "type": "string", "description": "User text input"}
        ],
        handler=text_input_handler,
    )
    rtvi.register_action(text_input_action)
    logger.info("Text input RTVI action registered")

    # Store the current client websocket reference for config sending (must be declared early)
    _current_websocket = None

    # ==========================================================================
    # JSON Message Handler for CLI Text Clients
    # ==========================================================================
    # This handler processes JSON messages from CLI clients that bypass the
    # Protobuf serialization layer. It handles RTVI actions like text_input:send.
    async def handle_json_message(message: str):
        """Handle JSON messages from CLI text clients.

        Processes RTVI protocol messages that come as JSON strings instead of
        binary Protobuf frames. This allows simple CLI clients to send text
        without needing the full pipecat-ai client library.

        Args:
            message: JSON string containing RTVI message
        """
        import json as json_module
        import uuid

        try:
            data = json_module.loads(message)
            msg_type = data.get("type", "")
            msg_id = data.get("id", str(uuid.uuid4()))

            logger.debug(f"[JSON Handler] Received: type={msg_type}")

            # Handle client-ready message
            if msg_type == "client-ready":
                logger.info("[JSON Handler] CLI client ready")
                # Trigger bot ready
                await rtvi.set_bot_ready()

                # Send server config to CLI client
                nonlocal _current_websocket
                if _current_websocket and _is_websocket_open(_current_websocket):
                    config_response = {
                        "id": str(uuid.uuid4()),
                        "type": "server-config",
                        "data": {
                            "service_mode": SERVICE_MODE,
                            "input_mode": "text" if not STT_ENABLED else "voice",
                            "stt": {"enabled": STT_ENABLED},
                            "llm": {"enabled": LLM_ENABLED, "model": str(server_config.llm.get("model", ""))},
                            "tts": {"enabled": TTS_ENABLED, "type": str(server_config.tts.get("type", ""))},
                        }
                    }
                    try:
                        await _current_websocket.send(json_module.dumps(config_response))
                        logger.info("[JSON Handler] Sent server config to CLI client")
                    except Exception as e:
                        # Gracefully handle connection closed during send
                        if "ConnectionClosed" in str(type(e).__name__):
                            logger.debug(f"[JSON Handler] Connection closed during config send: {e}")
                        else:
                            logger.warning(f"[JSON Handler] Error sending config: {e}")

            # Handle action requests (e.g., text_input:send)
            elif msg_type == "action":
                action_data = data.get("data", {})
                service = action_data.get("service", "")
                action = action_data.get("action", "")
                arguments = action_data.get("arguments", [])

                # Debug: Log full action data to diagnose argument format issues
                logger.info(f"[JSON Handler] Action: {service}:{action}")
                logger.debug(f"[JSON Handler] Full action_data: {action_data}")
                logger.debug(f"[JSON Handler] Arguments: {arguments}")

                if service == "text_input" and action == "send":
                    # Call text input handler directly
                    result = await text_input_handler(rtvi, service, arguments)

                    # Send response back to CLI client
                    if _current_websocket and _is_websocket_open(_current_websocket):
                        response = {
                            "id": msg_id,
                            "type": "action-response",
                            "data": {"result": result}
                        }
                        try:
                            await _current_websocket.send(json_module.dumps(response))
                            logger.debug(f"[JSON Handler] Sent action response: {result}")
                        except Exception as e:
                            # Gracefully handle connection closed during send
                            if "ConnectionClosed" in str(type(e).__name__):
                                logger.debug(f"[JSON Handler] Connection closed during response send: {e}")
                            else:
                                logger.warning(f"[JSON Handler] Error sending response: {e}")

                elif service == "config" and action == "get_server_config":
                    await get_server_config_handler(rtvi, service, arguments)

            else:
                logger.debug(f"[JSON Handler] Unhandled message type: {msg_type}")

        except json_module.JSONDecodeError as e:
            logger.warning(f"[JSON Handler] Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"[JSON Handler] Error: {e}")
            import traceback
            traceback.print_exc()

    # Register the JSON message handler with the WebSocket transport
    ws_transport.set_json_message_handler(handle_json_message)
    logger.info("JSON message handler registered for CLI text clients")

    # Register get_server_config action for client fallback requests
    async def get_server_config_handler(rtvi_processor: RTVIProcessor, service: str, arguments: dict[str, any]) -> bool:
        """Handle client request for server config (fallback mechanism)."""
        nonlocal _current_websocket
        import json
        import uuid

        logger.info("[CONFIG] Fallback: Client requested server config")

        try:
            config_data = {
                "service_mode": SERVICE_MODE,  # "full", "stt_llm", "stt_only", "llm_only", "tts_only", "llm_tts"
                "stt": {
                    "model": str(server_config.stt.get("model", "")) if STT_ENABLED else "",
                    "device": str(server_config.stt.get("device", "cpu")) if STT_ENABLED else "",
                    "type": str(server_config.stt.get("type", "nemo")) if STT_ENABLED else "",
                    "enabled": STT_ENABLED,
                },
                "llm": {
                    "model": str(server_config.llm.get("model", "")) if LLM_ENABLED else "",
                    "type": str(server_config.llm.get("type", "")) if LLM_ENABLED else "",
                    "enabled": LLM_ENABLED,
                },
                "tts": {
                    "type": str(server_config.tts.get("type", "")) if TTS_ENABLED else "",
                    "model": str(server_config.tts.get("model", "")) if TTS_ENABLED else "",
                    "enabled": TTS_ENABLED,
                },
                "diar": {
                    "enabled": bool(server_config.diar.get("enabled", False)) if STT_ENABLED else False,
                    "model": str(server_config.diar.get("model", "")) if STT_ENABLED else "",
                    "threshold": float(server_config.diar.get("threshold", 0.5)),
                },
                # Input mode indicator for UI
                "input_mode": "voice" if STT_ENABLED else "text",
            }

            # Create RTVI-compatible message format
            message = {
                "id": str(uuid.uuid4()),
                "type": "server-config",
                "data": config_data,
            }
            message_json = json.dumps(message)

            logger.info(f"[CONFIG] Fallback sending: Mode={config_data['service_mode']}, Input={config_data['input_mode']}")
            logger.info(f"[CONFIG] Fallback sending: STT={config_data['stt']['enabled']}, LLM={config_data['llm']['enabled']}, TTS={config_data['tts']['enabled']}")

            # Try multiple methods to get the WebSocket
            websocket = None

            # Method 1: Use stored websocket reference
            if _is_websocket_open(_current_websocket):
                websocket = _current_websocket
                logger.info("[CONFIG] Using stored websocket reference")

            # Method 2: Get from input transport
            if not websocket:
                try:
                    ws_from_input = ws_transport._input._websocket if ws_transport._input else None
                    if _is_websocket_open(ws_from_input):
                        websocket = ws_from_input
                        logger.info("[CONFIG] Using ws_transport._input._websocket")
                except Exception as e:
                    logger.debug(f"[CONFIG] Failed to get from input transport: {e}")

            # Method 3: Get from transport directly
            if not websocket:
                try:
                    ws_from_transport = getattr(ws_transport, '_websocket', None)
                    if _is_websocket_open(ws_from_transport):
                        websocket = ws_from_transport
                        logger.info("[CONFIG] Using ws_transport._websocket")
                except Exception as e:
                    logger.debug(f"[CONFIG] Failed to get from transport: {e}")

            if websocket:
                await websocket.send(message_json)
                logger.info("[CONFIG] ✓ Fallback config sent successfully")
                return True
            else:
                logger.warning("[CONFIG] ✗ WebSocket not available for fallback config send")
                return False
        except Exception as e:
            logger.error(f"[CONFIG] ✗ Error sending fallback config: {e}")
            import traceback
            traceback.print_exc()
            return False

    get_config_action = RTVIAction(
        service="config",
        action="get_server_config",
        result="bool",
        arguments=[],
        handler=get_server_config_handler,
    )
    rtvi.register_action(get_config_action)

    logger.info("Setting up pipeline...")

    # Initialize realtime display for clean console output
    voice_display = get_voice_agent_display()
    display_processor = VoiceAgentDisplayProcessor(
        display=voice_display,
        show_stt=True,
        show_llm=LLM_ENABLED,  # Only show LLM output when enabled
        show_tts=False,  # TTS display can be verbose, disabled by default
    )

    # =============================================================================
    # Pipeline Construction based on Service Mode
    # =============================================================================
    #
    # Voice Input Modes (STT enabled):
    # - Full mode (STT+LLM+TTS):  Input → RTVI → STT → [Diar] → TurnTaking → Display → UserContext → LLM → TTS → Output → AssistantContext
    # - STT+LLM mode (no TTS):    Input → RTVI → STT → [Diar] → TurnTaking → Display → UserContext → LLM → Output → AssistantContext
    # - STT-only mode (no LLM):   Input → STT → [Diar] → Aggregator → [SpeakerFrame] → RTVI → Output
    #
    # Text Input Modes (STT disabled):
    # - LLM+TTS mode:             Input → RTVI → Display → UserContext → LLM → TTS → Output → AssistantContext
    # - LLM-only mode:            Input → RTVI → Display → UserContext → LLM → Output → AssistantContext
    # - TTS-only mode:            Input → RTVI → TTS → Output

    pipeline = [
        ws_transport.input(),
    ]

    if STT_ENABLED:
        # =========================================================================
        # Voice Input Modes (STT enabled)
        # =========================================================================
        if LLM_ENABLED:
            # Full mode or STT+LLM mode: Use turn-taking and LLM pipeline
            pipeline.append(rtvi)
            pipeline.append(stt)

            if USE_DIAR and diar is not None:
                pipeline.append(diar)

            # Add turn-taking, display, and context components
            pipeline.extend([turn_taking, display_processor, user_context_aggregator, llm])

            if TTS_ENABLED and tts is not None:
                pipeline.append(tts)

            pipeline.append(ws_transport.output())
            pipeline.append(assistant_context_aggregator)

            logger.info(f"Pipeline configured for {SERVICE_MODE} mode: Input → RTVI → STT → {'Diar → ' if USE_DIAR else ''}TurnTaking → Display → LLM → {'TTS → ' if TTS_ENABLED else ''}Output")
        else:
            # STT-only mode: Use aggregator pipeline without LLM/TTS
            # Note: RTVIObserver will be created after pipeline, so we create a placeholder
            # SpeakerFrameProcessor for STT-only mode with diarization
            pipeline.append(stt)

            if USE_DIAR and diar is not None:
                pipeline.append(diar)
                if speaker_aggregator is not None:
                    pipeline.append(speaker_aggregator)
                    # Create speaker frame processor - will be initialized with rtvi_observer later
                    # For now, pass None and it will use direct RTVI processing
                    speaker_frame_processor = SpeakerFrameProcessor(None)
                    pipeline.append(speaker_frame_processor)
            else:
                if stt_aggregator is not None:
                    pipeline.append(stt_aggregator)

            pipeline.append(rtvi)
            pipeline.append(ws_transport.output())

            logger.info(f"Pipeline configured for {SERVICE_MODE} mode: Input → STT → {'Diar → SpeakerAggregator → ' if USE_DIAR else 'STTAggregator → '}RTVI → Output")
    else:
        # =========================================================================
        # Text Input Modes (STT disabled)
        # =========================================================================
        # No VAD, STT, Turn-Taking, or Diarization needed!
        # Input is handled via text_input RTVI Action (Enter key = end of turn)

        pipeline.append(rtvi)  # RTVI Action handles text input

        if LLM_ENABLED:
            # LLM-Only or LLM+TTS mode: Text input → LLM → [TTS] → Output
            pipeline.extend([display_processor, user_context_aggregator, llm])

            if TTS_ENABLED and tts is not None:
                pipeline.append(tts)

            pipeline.append(ws_transport.output())
            pipeline.append(assistant_context_aggregator)

            if TTS_ENABLED:
                logger.info(f"Pipeline configured for {SERVICE_MODE} mode: Input → RTVI → Display → LLM → TTS → Output")
            else:
                logger.info(f"Pipeline configured for {SERVICE_MODE} mode: Input → RTVI → Display → LLM → Output")

        elif TTS_ENABLED and tts is not None:
            # TTS-Only mode: Text input → TTS → Output (no LLM processing)
            pipeline.append(tts)
            pipeline.append(ws_transport.output())

            logger.info(f"Pipeline configured for {SERVICE_MODE} mode: Input → RTVI → TTS → Output")

        else:
            # No processing available (invalid configuration)
            raise ValueError("At least one of STT, LLM, or TTS must be enabled")

    pipeline = Pipeline(pipeline)

    rtvi_text_aggregator = SimpleSegmentedTextAggregator(punctuation_marks=".!?\n")
    rtvi_observer = RTVIObserver(rtvi, text_aggregator=rtvi_text_aggregator)

    # Configure TTS enabled flag for LLM-Only mode handling
    # When TTS is disabled, RTVIObserver needs to send 'idle' status when LLM completes
    rtvi_observer.set_tts_enabled(TTS_ENABLED)

    # Link RTVIObserver to SpeakerFrameProcessor if in STT-only mode with diarization
    if not LLM_ENABLED and speaker_frame_processor is not None:
        speaker_frame_processor._rtvi_observer = rtvi_observer
        logger.debug("Linked RTVIObserver to SpeakerFrameProcessor for STT-only mode")
    # Interruptions only needed in conversation modes (LLM enabled)
    allow_interruptions = LLM_ENABLED

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=allow_interruptions,
            enable_metrics=False,
            enable_usage_metrics=False,
            send_initial_empty_metrics=True,
            report_only_initial_ttfb=True,
            idle_timeout=None,  # Disable idle timeout
        ),
        observers=[rtvi_observer],
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )

    # Track task state
    task_running = True

    # Setup logging again to avoid logger from being overwritten during setting up the pipeline components
    setup_logging()

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi: RTVIProcessor):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()

        # Display clean header for voice agent session
        mode_display = {
            # Voice input modes
            "full": "Voice Agent Active (Full Mode)",
            "stt_llm": "Voice Agent Active (STT+LLM Mode)",
            "stt_only": "STT-Only Mode Active",
            # Text input modes
            "llm_only": "Text Chat Mode Active (LLM-Only)",
            "tts_only": "TTS-Only Mode Active (Text→Speech)",
            "llm_tts": "Text Chat Mode Active (LLM+TTS)",
        }
        voice_display.header(f"[*] {mode_display.get(SERVICE_MODE, 'Voice Agent Active')}")

        # Send server configuration to client for model info display
        # Add a small delay to allow client to set up WebSocket interceptor
        # Then send config multiple times to ensure delivery
        import asyncio
        await asyncio.sleep(0.1)  # 100ms delay for client setup
        await send_server_config_direct()

        # Send again after 500ms as backup
        asyncio.create_task(send_config_with_delay(0.5))

        # Kick off the conversation (only in LLM-enabled voice input modes)
        # Text input modes wait for user to send first message via text_input action
        if STT_ENABLED and LLM_ENABLED and user_context_aggregator is not None:
            try:
                await task.queue_frames([user_context_aggregator.get_context_frame()])
            except Exception as e:
                logger.error(f"Error queuing context frame: {e}")
        elif not STT_ENABLED and LLM_ENABLED:
            logger.info("Text input mode: Waiting for user text input (use text_input:send action)")
        else:
            logger.info("STT-only mode: Skipping initial context frame (no LLM)")

    async def send_config_with_delay(delay: float):
        """Send config after a delay as backup."""
        import asyncio
        await asyncio.sleep(delay)
        await send_server_config_direct()

    async def send_server_config_direct():
        """Send server config directly through WebSocket for reliability."""
        nonlocal _current_websocket
        import json
        import uuid

        try:
            config_data = {
                "service_mode": SERVICE_MODE,  # "full", "stt_llm", "stt_only", "llm_only", "tts_only", "llm_tts"
                "stt": {
                    "model": str(server_config.stt.get("model", "")) if STT_ENABLED else "",
                    "device": str(server_config.stt.get("device", "cpu")) if STT_ENABLED else "",
                    "type": str(server_config.stt.get("type", "nemo")) if STT_ENABLED else "",
                    "enabled": STT_ENABLED,
                },
                "llm": {
                    "model": str(server_config.llm.get("model", "")) if LLM_ENABLED else "",
                    "type": str(server_config.llm.get("type", "")) if LLM_ENABLED else "",
                    "enabled": LLM_ENABLED,
                },
                "tts": {
                    "type": str(server_config.tts.get("type", "")) if TTS_ENABLED else "",
                    "model": str(server_config.tts.get("model", "")) if TTS_ENABLED else "",
                    "enabled": TTS_ENABLED,
                },
                "diar": {
                    "enabled": bool(server_config.diar.get("enabled", False)) if STT_ENABLED else False,
                    "model": str(server_config.diar.get("model", "")) if STT_ENABLED else "",
                    "threshold": float(server_config.diar.get("threshold", 0.5)),
                },
                # Input mode indicator for UI
                "input_mode": "voice" if STT_ENABLED else "text",
            }

            # Create RTVI-compatible message format
            message = {
                "id": str(uuid.uuid4()),
                "type": "server-config",
                "data": config_data,
            }
            message_json = json.dumps(message)

            # Debug logging
            logger.info(f"[CONFIG] Attempting to send server config:")
            logger.info(f"[CONFIG] Mode: {config_data['service_mode']}, Input: {config_data['input_mode']}")
            logger.info(f"[CONFIG] STT: enabled={config_data['stt']['enabled']}")
            logger.info(f"[CONFIG] LLM: enabled={config_data['llm']['enabled']}")
            logger.info(f"[CONFIG] TTS: enabled={config_data['tts']['enabled']}")

            # Try multiple methods to get the WebSocket
            websocket = None

            # Method 1: Use stored websocket reference
            if _is_websocket_open(_current_websocket):
                websocket = _current_websocket
                logger.info("[CONFIG] Using stored websocket reference")

            # Method 2: Get from input transport
            if not websocket:
                try:
                    ws_from_input = ws_transport._input._websocket if ws_transport._input else None
                    if _is_websocket_open(ws_from_input):
                        websocket = ws_from_input
                        logger.info("[CONFIG] Using ws_transport._input._websocket")
                except Exception as e:
                    logger.debug(f"[CONFIG] Failed to get from input transport: {e}")

            # Method 3: Get from transport directly
            if not websocket:
                try:
                    ws_from_transport = getattr(ws_transport, '_websocket', None)
                    if _is_websocket_open(ws_from_transport):
                        websocket = ws_from_transport
                        logger.info("[CONFIG] Using ws_transport._websocket")
                except Exception as e:
                    logger.debug(f"[CONFIG] Failed to get from transport: {e}")

            if websocket:
                await websocket.send(message_json)
                logger.info(f"[CONFIG] ✓ Server config sent successfully via WebSocket")
            else:
                # Fallback to RTVI observer method
                logger.warning("[CONFIG] WebSocket not available, trying RTVI observer method")
                await rtvi_observer.send_server_config(config_data)

        except Exception as e:
            logger.error(f"[CONFIG] ✗ Failed to send server config: {e}")
            import traceback
            traceback.print_exc()

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal _current_websocket
        logger.info(f"Pipecat Client connected from {client.remote_address}")
        # Store websocket reference for config sending
        _current_websocket = client
        logger.info(f"[CONFIG] Stored websocket reference: {client}")
        # Reset RTVI state for new connection
        rtvi._client_ready = False
        rtvi._bot_ready = False

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Pipecat Client disconnected from {client.remote_address}")

        # ==========================================================================
        # CRITICAL: Reset STT state to prevent stale tokens in next session
        # ==========================================================================
        # The STT service accumulates tokens across audio chunks for streaming.
        # If a client disconnects mid-stream (before EOU), these tokens persist
        # and contaminate the next client session.
        # We must explicitly reset the STT state on every client disconnect.
        if stt is not None:
            try:
                # Reset the low-level ASR service state (accumulated tokens + encoder cache)
                # NemoSTTService stores the ASR model in self._model (NemoStreamingASRService)
                if hasattr(stt, '_model') and stt._model is not None:
                    stt._model.reset_full()
                    logger.info("[STT] Reset streaming ASR state (tokens + encoder cache cleared)")
                # Reset any buffered text and prefix tracking in the STT service itself
                if hasattr(stt, '_last_boundary_text'):
                    stt._last_boundary_text = ""
                if hasattr(stt, '_committed_text_prefix'):
                    stt._committed_text_prefix = ""
                if hasattr(stt, '_committed_prefix_timestamp'):
                    stt._committed_prefix_timestamp = 0.0
                if hasattr(stt, '_last_max_words_reset_word_count'):
                    stt._last_max_words_reset_word_count = 0
                # Reset audio buffer
                if hasattr(stt, 'audio_buffer'):
                    stt.audio_buffer = []
                logger.info("[STT] Reset all STT service state for new session")
            except Exception as e:
                logger.warning(f"[STT] Error resetting STT state: {e}")

        # Reset STT aggregators if present (STT-only mode)
        if stt_aggregator is not None:
            try:
                stt_aggregator.reset()
                logger.info("[Aggregator] STT aggregator state reset")
            except Exception as e:
                logger.warning(f"[Aggregator] Error resetting STT aggregator: {e}")

        if speaker_aggregator is not None:
            try:
                speaker_aggregator.reset()
                logger.info("[Aggregator] Speaker aggregator state reset")
            except Exception as e:
                logger.warning(f"[Aggregator] Error resetting speaker aggregator: {e}")

        # Reset diarization state if present
        if diar is not None:
            try:
                diar.reset()
                logger.info("[Diar] Diarization state reset")
            except Exception as e:
                logger.warning(f"[Diar] Error resetting diarization: {e}")

        # Don't cancel the task immediately - let it handle the disconnection gracefully
        # The task will continue running and can accept new connections
        # Only send an EndTaskFrame to clean up the current session
        if task_running:
            try:
                await task.queue_frames([EndTaskFrame()])
            except Exception as e:
                # Don't log warnings for normal connection closures
                if "ConnectionClosedOK" not in str(e) and "1005" not in str(e):
                    logger.warning(f"Error sending EndTaskFrame: {e}")
                else:
                    logger.debug(f"Normal connection closure: {e}")

    @ws_transport.event_handler("on_session_timeout")
    async def on_session_timeout(transport, client):
        logger.info(f"Session timeout for {client.remote_address}")
        # Don't cancel the task - keep server running indefinitely
        logger.info("Session timeout occurred but keeping server running")
        # Note: With session_timeout=None, this handler should never be called

    logger.info("Starting pipeline runner...")

    try:
        runner = PipelineRunner()
        # Run the task until shutdown is requested
        await asyncio.wait_for(runner.run(task), timeout=None)  # No timeout - run indefinitely
    except asyncio.TimeoutError:
        logger.info("Pipeline runner timeout (should not happen with no timeout)")
    except Exception as e:
        logger.error(f"Pipeline runner error: {e}")
        task_running = False
    finally:
        logger.info("Pipeline runner stopped")


if __name__ == "__main__":
    asyncio.run(run_bot_websocket_server())
