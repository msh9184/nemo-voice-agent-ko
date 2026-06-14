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
STT-Only WebSocket Server with Speaker Diarization Support

A simplified version of the voice agent that performs:
- Speech-to-text (STT) using NeMo models
- Speaker diarization (optional) for multi-speaker identification
- Real-time transcription with speaker attribution

No LLM or TTS components are loaded, resulting in faster startup and lower resource usage.

Pipeline Architecture:
    When Diarization DISABLED: Input → STT → Aggregator → RTVI → Output
    When Diarization ENABLED:  Input → STT → Diar → SpeakerAwareAggregator → RTVI → Output
"""

import asyncio
import os
import signal
import sys
import uuid

from loguru import logger
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from pipecat.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIAction, RTVIConfig, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer

# Use NeMo's custom RTVIObserver and SpeakerFrameProcessor for speaker diarization support
from nemo.agents.voice_agent.pipecat.processors.frameworks.rtvi import RTVIObserver, SpeakerFrameProcessor

from nemo.agents.voice_agent.pipecat.services.nemo.diar import NemoDiarService, NeMoDiarInputParams
from nemo.agents.voice_agent.pipecat.services.nemo.stt import NemoSTTService, NeMoSTTInputParams
from nemo.agents.voice_agent.pipecat.services.nemo.stt_aggregator import STTTextAggregator
from nemo.agents.voice_agent.pipecat.services.nemo.speaker_aware_aggregator import SpeakerAwareAggregator
from nemo.agents.voice_agent.pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from nemo.agents.voice_agent.pipecat.config import get_config_manager

# Import RTVIMessage for custom message types
from pipecat.processors.frameworks.rtvi import RTVIMessage


# =========================================================================
# Custom RTVI Message Classes for Config Communication
# Pipecat 0.0.84 requires 'id' field in RTVIMessage
# =========================================================================

class ConfigResultData(BaseModel):
    """Data for config update result message."""
    success: bool
    succeeded: int = 0
    total: int = 0
    results: dict = Field(default_factory=dict)


class ConfigAvailableData(BaseModel):
    """Data for config available message."""
    config: dict
    specs: dict
    diarization_enabled: bool


class ConfigResetResultData(BaseModel):
    """Data for config reset result message."""
    success: bool
    succeeded: int = 0
    total: int = 0


class ConfigUpdateResultMessage(RTVIMessage):
    """RTVI message for config update result."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "config-update-result"
    data: ConfigResultData


class ConfigAvailableMessage(RTVIMessage):
    """RTVI message for config available."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "config-available"
    data: ConfigAvailableData


class ConfigResetResultMessage(RTVIMessage):
    """RTVI message for config reset result."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "config-reset-result"
    data: ConfigResetResultData


def setup_logging(level: str = "INFO"):
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR). Default is INFO for cleaner console output.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | <level>{message}</level>",
        level=level,
    )
    logger.add("stt_only_server.log", rotation="1 day", level="DEBUG")  # File keeps full debug


# Set LOG_LEVEL environment variable for verbose output: export LOG_LEVEL=DEBUG
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
setup_logging(LOG_LEVEL)

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()


def load_stt_only_config():
    """
    Load STT-only configuration without requiring LLM/TTS configs.
    This is a simplified config loader for STT-only mode.
    """
    server_base_path = os.path.dirname(__file__)
    config_path = os.environ.get(
        "SERVER_CONFIG_PATH",
        os.path.join(server_base_path, "server_configs", "stt_only_mode.yaml")
    )

    if not os.path.exists(config_path):
        logger.warning(f"Config not found at {config_path}, using defaults")
        return None

    logger.info(f"Loading STT-only config from: {config_path}")
    return OmegaConf.load(config_path)


# Load configuration
server_config = load_stt_only_config()

# Default constants
SAMPLE_RATE = 16000
RAW_AUDIO_FRAME_LEN_IN_SECS = 0.016

# Transport configuration
TRANSPORT_AUDIO_OUT_10MS_CHUNKS = 4
if server_config and hasattr(server_config, 'transport'):
    TRANSPORT_AUDIO_OUT_10MS_CHUNKS = server_config.transport.get('audio_out_10ms_chunks', 4)

# VAD configuration
vad_params = VADParams(
    confidence=0.6,
    start_secs=0.1,
    stop_secs=1.0,
    min_volume=0.4,
)
if server_config and hasattr(server_config, 'vad'):
    vad_params = VADParams(
        confidence=server_config.vad.get('confidence', 0.6),
        start_secs=server_config.vad.get('start_secs', 0.1),
        stop_secs=server_config.vad.get('stop_secs', 1.0),
        min_volume=server_config.vad.get('min_volume', 0.4),
    )

# STT configuration
# MODEL OPTIONS:
# 1. nvidia/parakeet_realtime_eou_120m-v1 - EOU detection (may have streaming issues in STT-only mode)
# 2. nvidia/stt_en_fastconformer_hybrid_large_streaming_multi - Recommended for STT-only
STT_MODEL_PATH = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"
STT_DEVICE = "cuda:0"
# Frame length for streaming (80ms is model's designed value - DO NOT CHANGE)
STT_FRAME_LEN = 0.08
STT_ATT_CONTEXT = [70, 1]

if server_config and hasattr(server_config, 'stt'):
    STT_MODEL_PATH = server_config.stt.get('model', STT_MODEL_PATH)
    STT_DEVICE = server_config.stt.get('device', STT_DEVICE)
    STT_FRAME_LEN = server_config.stt.get('frame_len_in_secs', STT_FRAME_LEN)
    att_ctx = server_config.stt.get('att_context_size', None)
    if att_ctx is not None:
        STT_ATT_CONTEXT = list(att_ctx) if hasattr(att_ctx, '__iter__') else STT_ATT_CONTEXT

stt_params = NeMoSTTInputParams(
    att_context_size=STT_ATT_CONTEXT,
    frame_len_in_secs=STT_FRAME_LEN,
    raw_audio_frame_len_in_secs=RAW_AUDIO_FRAME_LEN_IN_SECS,
)

# Diarization configuration (optional)
DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"
USE_DIAR = False
diar_params = NeMoDiarInputParams(
    frame_len_in_secs=0.08,
    threshold=0.4,
)
if server_config and hasattr(server_config, 'diar'):
    DIAR_MODEL = server_config.diar.get('model', DIAR_MODEL)
    USE_DIAR = server_config.diar.get('enabled', False)
    diar_params = NeMoDiarInputParams(
        frame_len_in_secs=server_config.diar.get('frame_len_in_secs', 0.08),
        threshold=server_config.diar.get('threshold', 0.4),
    )

# Aggregator configuration
AGGREGATOR_USE_VAD_BOUNDARY = True
AGGREGATOR_USE_PUNCTUATION_BOUNDARY = True
AGGREGATOR_MIN_WORDS = 3
AGGREGATOR_MAX_WORDS = 20
AGGREGATOR_PUNCTUATION = ".?!。？！"
AGGREGATOR_SILENCE_TIMEOUT = 3.0
AGGREGATOR_SILENCE_CHECK_INTERVAL = 0.5

if server_config and hasattr(server_config, 'aggregator'):
    AGGREGATOR_USE_VAD_BOUNDARY = server_config.aggregator.get('use_vad_boundary', True)
    AGGREGATOR_USE_PUNCTUATION_BOUNDARY = server_config.aggregator.get('use_punctuation_boundary', True)
    AGGREGATOR_MIN_WORDS = server_config.aggregator.get('min_words_for_punctuation_boundary', 3)
    AGGREGATOR_MAX_WORDS = server_config.aggregator.get('max_words_before_commit', 20)
    AGGREGATOR_PUNCTUATION = server_config.aggregator.get('sentence_end_punctuation', ".?!。？！")
    AGGREGATOR_SILENCE_TIMEOUT = server_config.aggregator.get('silence_timeout_secs', 3.0)
    AGGREGATOR_SILENCE_CHECK_INTERVAL = server_config.aggregator.get('silence_check_interval_secs', 0.5)

# Speaker-aware aggregator configuration (used when diarization is enabled)
SPEAKER_MAX_SPEAKERS = 4
SPEAKER_USE_BOUNDARY = True
SPEAKER_MIN_WORDS_FOR_COMMIT = 1
SPEAKER_CONSOLE_DISPLAY = True

if server_config and hasattr(server_config, 'speaker_aggregator'):
    SPEAKER_MAX_SPEAKERS = server_config.speaker_aggregator.get('max_speakers', 4)
    SPEAKER_USE_BOUNDARY = server_config.speaker_aggregator.get('use_speaker_boundary', True)
    SPEAKER_MIN_WORDS_FOR_COMMIT = server_config.speaker_aggregator.get('min_words_for_speaker_commit', 1)
    SPEAKER_CONSOLE_DISPLAY = server_config.speaker_aggregator.get('console_display', True)

logger.info(f"STT-Only Server Configuration:")
logger.info(f"  STT Model: {STT_MODEL_PATH}")
logger.info(f"  STT Device: {STT_DEVICE}")
logger.info(f"  Diarization Enabled: {USE_DIAR}")
if USE_DIAR:
    logger.info(f"  Diarization Model: {DIAR_MODEL}")
    logger.info(f"  Speaker Max Speakers: {SPEAKER_MAX_SPEAKERS}")
    logger.info(f"  Speaker Boundary Enabled: {SPEAKER_USE_BOUNDARY}")
    logger.info(f"  Speaker Console Display: {SPEAKER_CONSOLE_DISPLAY}")
logger.info(f"  Aggregator VAD Boundary: {AGGREGATOR_USE_VAD_BOUNDARY}")
logger.info(f"  Aggregator Punctuation Boundary: {AGGREGATOR_USE_PUNCTUATION_BOUNDARY}")
logger.info(f"  Aggregator Max Words: {AGGREGATOR_MAX_WORDS}")
logger.info(f"  Aggregator Silence Timeout: {AGGREGATOR_SILENCE_TIMEOUT}s")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


async def run_stt_only_websocket_server(host: str = "0.0.0.0", port: int = 8765):
    """
    Run STT-only WebSocket server without LLM and TTS components.

    This server:
    - Receives audio via WebSocket
    - Performs streaming ASR using NeMo models
    - Sends transcription results back to client
    - Optionally performs speaker diarization
    """
    logger.info(f"Starting STT-only WebSocket server on {host}:{port}")
    logger.info("Server configured to run indefinitely, use Ctrl+C to quit.")
    logger.info("=" * 60)
    logger.info("STT-ONLY MODE: No LLM or TTS components loaded")
    logger.info("=" * 60)

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize VAD
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=vad_params,
    )
    logger.info("VAD analyzer initialized")

    # Initialize WebSocket transport
    ws_transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            serializer=ProtobufFrameSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=False,  # No audio output needed for STT-only
            add_wav_header=False,
            vad_analyzer=vad_analyzer,
            session_timeout=None,
            audio_in_sample_rate=SAMPLE_RATE,
            can_create_user_frames=True,  # Allow VAD to create frames
            audio_out_10ms_chunks=TRANSPORT_AUDIO_OUT_10MS_CHUNKS,
        ),
        host=host,
        port=port,
    )
    logger.info("WebSocket transport initialized")

    # Initialize STT service
    logger.info(f"Loading STT model: {STT_MODEL_PATH}")
    stt = NemoSTTService(
        model=STT_MODEL_PATH,
        device=STT_DEVICE,
        params=stt_params,
        sample_rate=SAMPLE_RATE,
        audio_passthrough=USE_DIAR,  # Pass audio to diarization when enabled
        has_turn_taking=False,  # No turn-taking in STT-only mode
        backend="legacy",
        decoder_type="rnnt",
    )
    logger.info("STT service initialized")

    # Initialize Diarization (optional)
    diar = None
    if USE_DIAR:
        logger.info(f"Loading Diarization model: {DIAR_MODEL}")
        diar = NemoDiarService(
            model=DIAR_MODEL,
            device=STT_DEVICE,
            params=diar_params,
            sample_rate=SAMPLE_RATE,
            backend="legacy",
            enabled=USE_DIAR,
        )
        logger.info("Diarization service initialized")

    # RTVI processor for client communication
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    # Create RTVIObserver instance (NeMo custom) for speaker diarization frame support
    rtvi_observer = RTVIObserver(rtvi)

    # Build pipeline based on diarization mode
    # When Diarization DISABLED: Input → STT → STTTextAggregator → RTVI → Output
    # When Diarization ENABLED:  Input → STT → Diar → SpeakerAwareAggregator → RTVI → Output
    logger.info("Building STT-only pipeline...")

    if USE_DIAR and diar is not None:
        # Speaker-aware aggregator for multi-speaker scenarios
        # Handles speaker change detection in addition to VAD/punctuation boundaries
        speaker_aggregator = SpeakerAwareAggregator(
            max_speakers=SPEAKER_MAX_SPEAKERS,
            use_speaker_boundary=SPEAKER_USE_BOUNDARY,
            use_vad_boundary=AGGREGATOR_USE_VAD_BOUNDARY,
            use_punctuation_boundary=AGGREGATOR_USE_PUNCTUATION_BOUNDARY,
            min_words_for_speaker_commit=SPEAKER_MIN_WORDS_FOR_COMMIT,
            min_words_for_punctuation_boundary=AGGREGATOR_MIN_WORDS,
            max_words_before_commit=AGGREGATOR_MAX_WORDS,
            sentence_end_punctuation=AGGREGATOR_PUNCTUATION,
            silence_timeout_secs=AGGREGATOR_SILENCE_TIMEOUT,
            silence_check_interval_secs=AGGREGATOR_SILENCE_CHECK_INTERVAL,
            eou_string="<EOU>",
            eob_string="<EOB>",
            use_realtime_display=SPEAKER_CONSOLE_DISPLAY,  # Fixed: was 'console_display'
        )
        logger.info("Speaker-Aware Aggregator initialized")

        # Create SpeakerFrameProcessor to convert speaker frames to RTVI messages
        # NOTE: Must pass rtvi_observer (not rtvi) for push_transport_message_urgent to work
        speaker_frame_processor = SpeakerFrameProcessor(rtvi_observer)

        # Pipeline with diarization: STT → Diar → SpeakerAwareAggregator → SpeakerFrameProcessor
        pipeline_components = [
            ws_transport.input(),
            stt,
            diar,  # Diarization after STT (receives audio via passthrough)
            speaker_aggregator,  # Speaker-aware aggregation
            speaker_frame_processor,  # Convert speaker frames to RTVI messages
            rtvi,
            ws_transport.output(),
        ]
        logger.info("Pipeline: Input → STT → Diar → SpeakerAwareAggregator → SpeakerFrameProcessor → RTVI → Output")
    else:
        # Standard STT-only aggregator (no speaker tracking)
        stt_aggregator = STTTextAggregator(
            eou_string="<EOU>",
            eob_string="<EOB>",
            sentence_end_punctuation=AGGREGATOR_PUNCTUATION,
            use_vad_boundary=AGGREGATOR_USE_VAD_BOUNDARY,
            use_punctuation_boundary=AGGREGATOR_USE_PUNCTUATION_BOUNDARY,
            min_words_for_punctuation_boundary=AGGREGATOR_MIN_WORDS,
            max_words_before_commit=AGGREGATOR_MAX_WORDS,
            silence_timeout_secs=AGGREGATOR_SILENCE_TIMEOUT,
            silence_check_interval_secs=AGGREGATOR_SILENCE_CHECK_INTERVAL,
        )
        logger.info("STT Text Aggregator initialized")

        # Pipeline without diarization: STT → STTTextAggregator
        pipeline_components = [
            ws_transport.input(),
            stt,
            stt_aggregator,
            rtvi,
            ws_transport.output(),
        ]
        logger.info("Pipeline: Input → STT → STTTextAggregator → RTVI → Output")

    pipeline = Pipeline(pipeline_components)

    # Create pipeline task with RTVIObserver to handle transcription frames
    # The RTVIObserver converts TranscriptionFrame/InterimTranscriptionFrame to RTVI messages
    # Uses rtvi_observer which is NeMo's custom RTVIObserver with SpeakerTranscription support
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,  # No interruptions in STT-only mode
            enable_metrics=False,
            enable_usage_metrics=False,
            send_initial_empty_metrics=True,
            report_only_initial_ttfb=True,
            idle_timeout=None,
        ),
        observers=[rtvi_observer],  # IMPORTANT: Uses NeMo custom RTVIObserver for speaker frames
        idle_timeout_secs=None,
        cancel_on_idle_timeout=False,
    )

    # Initialize Runtime Config Manager
    config_manager = get_config_manager()

    # Register components for runtime configuration
    config_manager.register_component("vad", vad_analyzer)
    config_manager.register_component("stt", stt)
    if USE_DIAR and diar is not None:
        config_manager.register_component("diar", diar)
        config_manager.register_component("aggregator", speaker_aggregator)
        config_manager.register_component("speaker_aggregator", speaker_aggregator)
    else:
        config_manager.register_component("aggregator", stt_aggregator)

    # Set initial values from server config
    config_manager._current_config["vad.confidence"] = vad_params.confidence
    config_manager._current_config["vad.stop_secs"] = vad_params.stop_secs
    config_manager._current_config["vad.start_secs"] = vad_params.start_secs
    config_manager._current_config["vad.min_volume"] = vad_params.min_volume
    config_manager._current_config["aggregator.max_words"] = AGGREGATOR_MAX_WORDS
    config_manager._current_config["aggregator.silence_timeout"] = AGGREGATOR_SILENCE_TIMEOUT
    config_manager._current_config["aggregator.min_words"] = AGGREGATOR_MIN_WORDS
    config_manager._current_config["stt.att_context"] = list(STT_ATT_CONTEXT)
    config_manager._current_config["diar.max_speakers"] = SPEAKER_MAX_SPEAKERS

    logger.info("RuntimeConfigManager initialized with components")

    task_running = True

    setup_logging(LOG_LEVEL)

    # =====================================================================
    # Helper function for parsing RTVI arguments
    # =====================================================================
    def get_argument(args, name, default=None):
        """Extract argument value from RTVI arguments array.

        RTVI sends arguments as array of {name, value} objects.
        """
        if isinstance(args, list):
            for arg in args:
                if isinstance(arg, dict) and arg.get('name') == name:
                    return arg.get('value', default)
        elif isinstance(args, dict):
            return args.get(name, default)
        return default

    # =====================================================================
    # RTVIAction Handlers for Runtime Configuration
    # =====================================================================

    async def config_update_handler(
        rtvi_processor: RTVIProcessor, service: str, arguments: dict
    ) -> dict:
        """Handle config update action."""
        try:
            params = get_argument(arguments, 'params', {})
            result = config_manager.apply_config(params)
            logger.info(f"Config update result: {result}")

            # Send custom message with detailed result (uses top-level defined class)
            message = ConfigUpdateResultMessage(
                data=ConfigResultData(
                    success=result.get('success', False),
                    succeeded=result.get('succeeded', 0),
                    total=result.get('total', 0),
                    results=result.get('results', {})
                )
            )
            await rtvi_observer.push_transport_message_urgent(message)
            return {"success": result.get('success', False)}
        except Exception as e:
            logger.error(f"Config update error: {e}")
            return {"success": False, "error": str(e)}

    async def config_get_handler(
        rtvi_processor: RTVIProcessor, service: str, arguments: dict
    ) -> dict:
        """Handle config get action."""
        try:
            config_data = config_manager.get_current_config()
            specs = config_manager.get_all_specs()

            # Send custom message (uses top-level defined class)
            message = ConfigAvailableMessage(
                data=ConfigAvailableData(
                    config=config_data,
                    specs=specs,
                    diarization_enabled=USE_DIAR
                )
            )
            await rtvi_observer.push_transport_message_urgent(message)
            logger.debug("Sent config-available message")
            return {"success": True}
        except Exception as e:
            logger.error(f"Config get error: {e}")
            return {"success": False, "error": str(e)}

    async def config_reset_handler(
        rtvi_processor: RTVIProcessor, service: str, arguments: dict
    ) -> dict:
        """Handle config reset action."""
        try:
            defaults = {}
            for name, spec in config_manager.PARAM_SPECS.items():
                defaults[name] = spec.default
            result = config_manager.apply_config(defaults)

            # Send custom message (uses top-level defined class)
            message = ConfigResetResultMessage(
                data=ConfigResetResultData(
                    success=result.get('success', False),
                    succeeded=result.get('succeeded', 0),
                    total=result.get('total', 0)
                )
            )
            await rtvi_observer.push_transport_message_urgent(message)
            logger.info("Config reset to defaults")
            return {"success": True}
        except Exception as e:
            logger.error(f"Config reset error: {e}")
            return {"success": False, "error": str(e)}

    # =====================================================================
    # Register RTVIActions for config service
    # Note: Pipecat 0.0.84 RTVIAction types: 'bool', 'number', 'string', 'array', 'object'
    # =====================================================================
    config_update_action = RTVIAction(
        service="config",
        action="update",
        result="object",  # Pipecat uses 'object' not 'dict'
        arguments=[{"name": "params", "type": "object"}],  # Pipecat uses 'object' not 'dict'
        handler=config_update_handler,
    )
    rtvi.register_action(config_update_action)
    logger.debug("Registered config:update action")

    config_get_action = RTVIAction(
        service="config",
        action="get",
        result="object",  # Pipecat uses 'object' not 'dict'
        arguments=[],
        handler=config_get_handler,
    )
    rtvi.register_action(config_get_action)
    logger.debug("Registered config:get action")

    config_reset_action = RTVIAction(
        service="config",
        action="reset",
        result="object",  # Pipecat uses 'object' not 'dict'
        arguments=[],
        handler=config_reset_handler,
    )
    rtvi.register_action(config_reset_action)
    logger.debug("Registered config:reset action")

    # =====================================================================
    # Event Handlers
    # =====================================================================

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi: RTVIProcessor):
        logger.info("RTVI Client ready - STT-only mode")
        await rtvi.set_bot_ready()
        # Send initial config to client (uses top-level defined class)
        try:
            config_data = config_manager.get_current_config()
            specs = config_manager.get_all_specs()

            message = ConfigAvailableMessage(
                data=ConfigAvailableData(
                    config=config_data,
                    specs=specs,
                    diarization_enabled=USE_DIAR
                )
            )
            await rtvi_observer.push_transport_message_urgent(message)
            logger.debug("Sent initial config to client")
        except Exception as e:
            logger.warning(f"Failed to send initial config: {e}")

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected from {client.remote_address}")
        rtvi._client_ready = False
        rtvi._bot_ready = False

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected from {client.remote_address}")
        if task_running:
            try:
                await task.queue_frames([EndTaskFrame()])
            except Exception as e:
                if "ConnectionClosedOK" not in str(e) and "1005" not in str(e):
                    logger.warning(f"Error sending EndTaskFrame: {e}")

    logger.info("=" * 60)
    if USE_DIAR:
        logger.info("STT + Diarization Server Ready!")
        logger.info(f"Mode: Multi-speaker (up to {SPEAKER_MAX_SPEAKERS} speakers)")
    else:
        logger.info("STT-Only Server Ready!")
        logger.info("Mode: Single-speaker")
    logger.info(f"WebSocket endpoint: ws://{host}:{port}")
    logger.info("=" * 60)

    try:
        runner = PipelineRunner()
        await asyncio.wait_for(runner.run(task), timeout=None)
    except asyncio.TimeoutError:
        logger.info("Pipeline runner timeout (should not happen with no timeout)")
    except Exception as e:
        logger.error(f"Pipeline runner error: {e}")
        task_running = False
    finally:
        logger.info("Pipeline runner stopped")


if __name__ == "__main__":
    asyncio.run(run_stt_only_websocket_server())
