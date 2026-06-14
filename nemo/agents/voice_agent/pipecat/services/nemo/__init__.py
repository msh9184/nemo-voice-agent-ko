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
NeMo Pipecat Services Package

This module uses lazy imports to allow importing individual services
without loading heavy dependencies (like vLLM for LLM service).
"""


def __getattr__(name):
    """Lazy import for NeMo services to avoid loading unnecessary dependencies."""
    if name == "NemoDiarService":
        from .diar import NemoDiarService
        return NemoDiarService
    elif name == "HuggingFaceLLMService":
        from .llm import HuggingFaceLLMService
        return HuggingFaceLLMService
    elif name == "NemoSTTService":
        from .stt import NemoSTTService
        return NemoSTTService
    elif name == "NeMoFastPitchHiFiGANTTSService":
        from .tts import NeMoFastPitchHiFiGANTTSService
        return NeMoFastPitchHiFiGANTTSService
    elif name == "FishSpeechService":
        from .tts import FishSpeechService
        return FishSpeechService
    elif name == "FishSpeechAPIService":
        from .tts import FishSpeechAPIService
        return FishSpeechAPIService
    elif name == "NeMoTurnTakingService":
        from .turn_taking import NeMoTurnTakingService
        return NeMoTurnTakingService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NemoDiarService",
    "HuggingFaceLLMService",
    "NemoSTTService",
    "NeMoFastPitchHiFiGANTTSService",
    "FishSpeechService",
    "FishSpeechAPIService",
    "NeMoTurnTakingService",
]
