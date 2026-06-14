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
Runtime Configuration Manager for NeMo Voice Agent.

Provides dynamic configuration changes during runtime without server restart.
Supports VAD, Aggregator, STT, and Diarization parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from loguru import logger


class ConfigCategory(Enum):
    """Configuration categories."""
    VAD = "vad"
    AGGREGATOR = "aggregator"
    STT = "stt"
    DIARIZATION = "diar"


@dataclass
class ParamSpec:
    """Parameter specification for validation."""
    name: str
    category: ConfigCategory
    param_type: type
    default: Any
    description: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    options: Optional[List[Any]] = None
    unit: str = ""

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value against this spec."""
        # Type check
        if self.options:
            # For options, check if value is in allowed list
            if value not in self.options:
                return False, f"Value must be one of {self.options}"
            return True, ""

        if self.param_type == bool:
            if not isinstance(value, bool):
                return False, f"Expected bool, got {type(value).__name__}"
            return True, ""

        if self.param_type in (int, float):
            try:
                val = self.param_type(value)
            except (ValueError, TypeError):
                return False, f"Cannot convert to {self.param_type.__name__}"

            if self.min_val is not None and val < self.min_val:
                return False, f"Value must be >= {self.min_val}"
            if self.max_val is not None and val > self.max_val:
                return False, f"Value must be <= {self.max_val}"
            return True, ""

        return True, ""


class RuntimeConfigManager:
    """
    Manages runtime configuration for Voice Agent components.

    Allows dynamic parameter changes via RTVI actions without server restart.
    """

    # Parameter specifications
    PARAM_SPECS: Dict[str, ParamSpec] = {
        # VAD Parameters
        "vad.confidence": ParamSpec(
            name="vad.confidence",
            category=ConfigCategory.VAD,
            param_type=float,
            default=0.6,
            min_val=0.0,
            max_val=1.0,
            description="Voice detection confidence threshold",
            unit=""
        ),
        "vad.stop_secs": ParamSpec(
            name="vad.stop_secs",
            category=ConfigCategory.VAD,
            param_type=float,
            default=1.0,
            min_val=0.1,
            max_val=5.0,
            description="Silence duration before speech end detection",
            unit="s"
        ),
        "vad.start_secs": ParamSpec(
            name="vad.start_secs",
            category=ConfigCategory.VAD,
            param_type=float,
            default=0.1,
            min_val=0.01,
            max_val=1.0,
            description="Speech duration before detection starts",
            unit="s"
        ),
        "vad.min_volume": ParamSpec(
            name="vad.min_volume",
            category=ConfigCategory.VAD,
            param_type=float,
            default=0.4,
            min_val=0.0,
            max_val=1.0,
            description="Minimum volume threshold for speech",
            unit=""
        ),

        # Aggregator Parameters
        "aggregator.max_words": ParamSpec(
            name="aggregator.max_words",
            category=ConfigCategory.AGGREGATOR,
            param_type=int,
            default=20,
            min_val=5,
            max_val=100,
            description="Maximum words before forced commit",
            unit="words"
        ),
        "aggregator.silence_timeout": ParamSpec(
            name="aggregator.silence_timeout",
            category=ConfigCategory.AGGREGATOR,
            param_type=float,
            default=3.0,
            min_val=0.5,
            max_val=30.0,
            description="Silence timeout for buffer flush",
            unit="s"
        ),
        "aggregator.min_words": ParamSpec(
            name="aggregator.min_words",
            category=ConfigCategory.AGGREGATOR,
            param_type=int,
            default=3,
            min_val=1,
            max_val=20,
            description="Minimum words for punctuation boundary",
            unit="words"
        ),
        "aggregator.console_display": ParamSpec(
            name="aggregator.console_display",
            category=ConfigCategory.AGGREGATOR,
            param_type=bool,
            default=True,
            description="Enable colored console output"
        ),

        # STT Parameters
        "stt.att_context": ParamSpec(
            name="stt.att_context",
            category=ConfigCategory.STT,
            param_type=list,
            default=[70, 1],
            options=[[70, 0], [70, 1], [70, 6], [70, 13]],
            description="Attention context size [lookback, lookahead]"
        ),

        # Diarization Parameters
        "diar.max_speakers": ParamSpec(
            name="diar.max_speakers",
            category=ConfigCategory.DIARIZATION,
            param_type=int,
            default=4,
            min_val=1,
            max_val=4,
            description="Maximum number of speakers to track",
            unit=""
        ),
        "diar.threshold": ParamSpec(
            name="diar.threshold",
            category=ConfigCategory.DIARIZATION,
            param_type=float,
            default=0.4,
            min_val=0.1,
            max_val=0.9,
            description="Speaker detection threshold",
            unit=""
        ),
    }

    def __init__(self):
        """Initialize the config manager."""
        self._current_config: Dict[str, Any] = {}
        self._components: Dict[str, Any] = {}
        self._change_callbacks: List[Callable] = []

        # Initialize with defaults
        for name, spec in self.PARAM_SPECS.items():
            self._current_config[name] = spec.default

        logger.info("RuntimeConfigManager initialized")

    def register_component(self, name: str, component: Any):
        """Register a component for configuration updates."""
        self._components[name] = component
        logger.debug(f"Registered component: {name}")

    def register_change_callback(self, callback: Callable):
        """Register a callback for configuration changes."""
        self._change_callbacks.append(callback)

    def get_param_spec(self, name: str) -> Optional[ParamSpec]:
        """Get parameter specification."""
        return self.PARAM_SPECS.get(name)

    def get_all_specs(self) -> Dict[str, Dict]:
        """Get all parameter specifications as dictionaries."""
        return {
            name: {
                "name": spec.name,
                "category": spec.category.value,
                "type": spec.param_type.__name__,
                "default": spec.default,
                "description": spec.description,
                "min": spec.min_val,
                "max": spec.max_val,
                "options": spec.options,
                "unit": spec.unit,
                "current": self._current_config.get(name, spec.default)
            }
            for name, spec in self.PARAM_SPECS.items()
        }

    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration values."""
        return dict(self._current_config)

    def validate_param(self, name: str, value: Any) -> tuple[bool, str]:
        """Validate a parameter value."""
        spec = self.PARAM_SPECS.get(name)
        if not spec:
            return False, f"Unknown parameter: {name}"
        return spec.validate(value)

    def apply_param(self, name: str, value: Any) -> Dict[str, Any]:
        """
        Apply a single parameter change.

        Returns:
            Dict with success status, message, and applied value
        """
        # Validate
        valid, msg = self.validate_param(name, value)
        if not valid:
            logger.warning(f"Config validation failed: {name}={value}: {msg}")
            return {"success": False, "error": msg, "param": name}

        spec = self.PARAM_SPECS[name]
        old_value = self._current_config.get(name)

        # Apply to component
        try:
            self._apply_to_component(name, value, spec)
            self._current_config[name] = value

            # Notify callbacks
            for callback in self._change_callbacks:
                try:
                    callback(name, value, old_value)
                except Exception as e:
                    logger.warning(f"Config change callback error: {e}")

            # Clean, compact log format
            logger.info(f"CFG  | {name}: {old_value} -> {value}")
            return {
                "success": True,
                "param": name,
                "value": value,
                "old_value": old_value
            }

        except Exception as e:
            logger.error(f"Failed to apply config {name}={value}: {e}")
            return {"success": False, "error": str(e), "param": name}

    def apply_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply multiple parameter changes.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            Dict with results for each parameter
        """
        results = {}
        for name, value in params.items():
            results[name] = self.apply_param(name, value)

        success_count = sum(1 for r in results.values() if r.get("success"))
        return {
            "success": success_count == len(params),
            "total": len(params),
            "succeeded": success_count,
            "results": results
        }

    def _apply_to_component(self, name: str, value: Any, spec: ParamSpec):
        """Apply configuration to the appropriate component."""
        category = spec.category

        if category == ConfigCategory.VAD:
            self._apply_vad_config(name, value)
        elif category == ConfigCategory.AGGREGATOR:
            self._apply_aggregator_config(name, value)
        elif category == ConfigCategory.STT:
            self._apply_stt_config(name, value)
        elif category == ConfigCategory.DIARIZATION:
            self._apply_diar_config(name, value)

    def _apply_vad_config(self, name: str, value: Any):
        """Apply VAD configuration."""
        vad = self._components.get("vad")
        if not vad:
            logger.debug("VAD component not registered, skipping apply")
            return

        param_name = name.split(".")[-1]  # e.g., "vad.confidence" -> "confidence"

        if hasattr(vad, "_params"):
            # SileroVADAnalyzer uses _params
            if hasattr(vad._params, param_name):
                setattr(vad._params, param_name, value)
                logger.debug(f"Applied VAD param: {param_name}={value}")
        elif hasattr(vad, "params"):
            if hasattr(vad.params, param_name):
                setattr(vad.params, param_name, value)
                logger.debug(f"Applied VAD param: {param_name}={value}")

    def _apply_aggregator_config(self, name: str, value: Any):
        """Apply Aggregator configuration."""
        aggregator = self._components.get("aggregator")
        if not aggregator:
            logger.debug("Aggregator component not registered, skipping apply")
            return

        param_name = name.split(".")[-1]

        # Map config names to attribute names
        attr_map = {
            "max_words": "max_words_before_commit",
            "silence_timeout": "silence_timeout_secs",
            "min_words": "min_words_for_punctuation_boundary",
            "console_display": "console_display",
        }

        attr_name = attr_map.get(param_name, param_name)

        if hasattr(aggregator, attr_name):
            setattr(aggregator, attr_name, value)
            logger.debug(f"Applied Aggregator param: {attr_name}={value}")

        # Also update speaker aggregator if exists
        speaker_agg = self._components.get("speaker_aggregator")
        if speaker_agg and hasattr(speaker_agg, attr_name):
            setattr(speaker_agg, attr_name, value)
            logger.debug(f"Applied SpeakerAggregator param: {attr_name}={value}")

    def _apply_stt_config(self, name: str, value: Any):
        """Apply STT configuration."""
        stt = self._components.get("stt")
        if not stt:
            logger.debug("STT component not registered, skipping apply")
            return

        param_name = name.split(".")[-1]

        if param_name == "att_context":
            # Apply attention context size change with verification
            if hasattr(stt, "_model") and hasattr(stt._model, "asr_model"):
                asr_model = stt._model.asr_model
                encoder = asr_model.encoder

                # Log current state before change
                old_context = getattr(stt._model, 'att_context_size', None)
                logger.info(f"STT  | att_context change: {old_context} -> {value}")

                if hasattr(encoder, "set_default_att_context_size"):
                    # Apply the change
                    encoder.set_default_att_context_size(att_context_size=value)

                    # Update internal state tracking
                    stt._model.att_context_size = value

                    # Verify the change was applied by checking encoder config
                    verified = False
                    if hasattr(encoder, 'att_context_size'):
                        actual = encoder.att_context_size
                        verified = (actual == value or str(actual) == str(value))
                        logger.info(f"STT  | Encoder att_context_size verified: {actual}")
                    elif hasattr(asr_model.cfg, 'encoder') and hasattr(asr_model.cfg.encoder, 'att_context_size'):
                        actual = asr_model.cfg.encoder.att_context_size
                        logger.info(f"STT  | Config att_context_size: {actual}")
                        verified = True

                    if not verified:
                        logger.warning(f"STT  | Cannot verify encoder att_context_size change")

                    # Reset state for clean start with new context
                    if hasattr(stt._model, "reset_state"):
                        stt._model.reset_state()
                        logger.info(f"STT  | State reset complete, new context active")
                else:
                    logger.error(f"STT  | Encoder missing set_default_att_context_size method")
            else:
                logger.error(f"STT  | Cannot access asr_model (stt._model.asr_model)")

    def _apply_diar_config(self, name: str, value: Any):
        """Apply Diarization configuration."""
        # Update speaker aggregator max_speakers
        speaker_agg = self._components.get("speaker_aggregator")
        if not speaker_agg:
            logger.debug("Speaker aggregator not registered, skipping apply")
            return

        param_name = name.split(".")[-1]

        if param_name == "max_speakers":
            if hasattr(speaker_agg, "max_speakers"):
                speaker_agg.max_speakers = value
                logger.debug(f"Applied max_speakers={value}")
        elif param_name == "threshold":
            diar = self._components.get("diar")
            if diar and hasattr(diar, "_params"):
                diar._params.threshold = value
                logger.debug(f"Applied diar threshold={value}")


# Global config manager instance
_config_manager: Optional[RuntimeConfigManager] = None


def get_config_manager() -> RuntimeConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = RuntimeConfigManager()
    return _config_manager


def reset_config_manager():
    """Reset the global config manager (mainly for testing)."""
    global _config_manager
    _config_manager = None
