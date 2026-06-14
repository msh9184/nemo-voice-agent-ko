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

This module uses lazy imports so individual services can be imported
without pulling in heavy dependencies (e.g. vLLM for the LLM service).
"""

# name -> (module, attribute)
_LAZY = {
    # ASR / STT
    "NemoSTTService": (".stt", "NemoSTTService"),
    # Diarization
    "NemoDiarService": (".diar", "NemoDiarService"),
    # LLM
    "HuggingFaceLLMService": (".llm", "HuggingFaceLLMService"),
    "HuggingFaceLLMLocalService": (".llm", "HuggingFaceLLMLocalService"),
    "VLLMService": (".llm", "VLLMService"),
    # TTS backends
    "NeMoFastPitchHiFiGANTTSService": (".tts", "NeMoFastPitchHiFiGANTTSService"),
    "KokoroTTSService": (".tts", "KokoroTTSService"),
    "MeloTTSKoreanService": (".tts", "MeloTTSKoreanService"),
    "CosyVoice3KoreanService": (".tts", "CosyVoice3KoreanService"),
    "CosyVoiceKoreanService": (".tts", "CosyVoiceKoreanService"),
    "FishSpeechKoreanService": (".tts", "FishSpeechKoreanService"),
    "FishSpeechAPIService": (".tts", "FishSpeechAPIService"),
    # Turn-taking
    "NeMoTurnTakingService": (".turn_taking", "NeMoTurnTakingService"),
}


def __getattr__(name):
    """Lazy import for NeMo services to avoid loading unnecessary dependencies."""
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        mod = importlib.import_module(module, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY.keys())
