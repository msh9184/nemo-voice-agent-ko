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
Custom frame types for NeMo Voice Agent.

Includes:
- DiarResultFrame: Speaker diarization results
- SpeakerTranscriptionFrame: Transcription with speaker information
- SpeakerStatusFrame: Real-time speaker status updates
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
from pipecat.frames.frames import DataFrame
from pipecat.transcriptions.language import Language


@dataclass
class DiarResultFrame(DataFrame):
    """Diarization result frame.

    Contains speaker identification results from the diarization model.

    Attributes:
        diar_result: Speaker ID (int) or raw probability array (ndarray)
        stream_id: Stream identifier for multi-stream scenarios
    """
    diar_result: np.ndarray | int
    stream_id: str = "default"


@dataclass
class SpeakerTranscriptionFrame(DataFrame):
    """Transcription frame with speaker information.

    Extends standard transcription with speaker identification
    for multi-speaker scenarios.

    Attributes:
        text: Transcribed text content
        speaker_id: Speaker identifier (0-3 for 4 speakers, None if unknown)
        is_final: Whether this is a final (committed) transcription
        reason: Commit reason (speaker_change, VAD, punctuation, max_words, timeout)
        timestamp: ISO 8601 timestamp
        language: Transcription language
    """
    text: str
    speaker_id: Optional[int] = None
    is_final: bool = False
    reason: str = ""
    timestamp: str = ""
    language: Language = Language.EN_US


@dataclass
class SpeakerInfo:
    """Information about a single speaker.

    Attributes:
        id: Speaker identifier (0-3)
        active: Whether speaker is currently speaking
        name: Display name (e.g., "Speaker 1")
        color: Color hex code for UI display
        activity_level: Activity intensity 0-100 for visualization
    """
    id: int
    active: bool = False
    name: str = ""
    color: str = ""
    activity_level: float = 0.0

    def __post_init__(self):
        if not self.name:
            self.name = f"Speaker {self.id + 1}"
        if not self.color:
            # Default speaker colors
            colors = ["#58a6ff", "#3fb950", "#d29922", "#a371f7"]
            self.color = colors[self.id % len(colors)]


@dataclass
class SpeakerStatusFrame(DataFrame):
    """Speaker status update frame.

    Provides real-time status of all speakers for UI updates.

    Attributes:
        speakers: List of SpeakerInfo for each speaker
        active_speaker_id: Currently active speaker ID (or None)
        total_speakers: Total number of speakers detected
    """
    speakers: List[SpeakerInfo] = field(default_factory=list)
    active_speaker_id: Optional[int] = None
    total_speakers: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "speakers": [
                {
                    "id": s.id,
                    "active": s.active,
                    "name": s.name,
                    "color": s.color,
                    "activity_level": s.activity_level
                }
                for s in self.speakers
            ],
            "active_speaker_id": self.active_speaker_id,
            "total_speakers": self.total_speakers
        }
