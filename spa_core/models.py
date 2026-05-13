from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


Region = Tuple[int, int]


@dataclass
class SpaSettings:
    speech_threshold_ms: float = 25.0
    pause_threshold_ms: float = 250.0
    sd_multiplier: float = 3.0
    threshold_mode: str = "automatic"  # automatic | manual
    adaptive: bool = True
    variable_mode: str = "ONE TIME"  # ONE TIME | PAUSE | SPEECH | AMPLITUDE
    iterations: int = 1
    amplitude_increment: float = 1.0
    time_increment_ms: float = 25.0
    plot_graph: bool = True
    use_full_file: bool = False
    use_fixed_segments: bool = False
    fixed_segment_seconds: float = 5.0
    passage_word_count: int = 97


@dataclass
class SpaSignal:
    path: Path
    raw_audio: np.ndarray
    sample_rate: int
    bit_depth: int
    detrended_audio: np.ndarray
    rectified_audio: np.ndarray
    filtered_envelope: np.ndarray
    normalized_envelope: np.ndarray
    analysis_region: Region = field(default_factory=lambda: (0, 0))
    noise_region: Optional[Region] = None
    manual_threshold: Optional[float] = None

    @property
    def analysis_start(self) -> int:
        return self.analysis_region[0]

    @property
    def analysis_end(self) -> int:
        return self.analysis_region[1]

    @property
    def analysis_audio(self) -> np.ndarray:
        return self.raw_audio[self.analysis_start : self.analysis_end]

    @property
    def analysis_envelope(self) -> np.ndarray:
        return self.normalized_envelope[self.analysis_start : self.analysis_end]


@dataclass
class SpaResult:
    settings: SpaSettings
    signal_path: Path
    sample_rate: int
    bit_depth: int
    analysis_region: Region
    noise_region: Optional[Region]
    threshold_curve: np.ndarray
    threshold_by_iteration: Dict[int, np.ndarray]
    speech_matrix: pd.DataFrame
    pause_matrix: pd.DataFrame
    total_matrix: pd.DataFrame
    speech_events_samples: np.ndarray
    pause_events_samples: np.ndarray
    fixed_events_samples: Optional[np.ndarray] = None
    segmentation_mode: str = "spa"  # spa | fixed

    @property
    def has_events(self) -> bool:
        has_fixed = self.fixed_events_samples is not None and len(self.fixed_events_samples) > 0
        has_spa_events = len(self.speech_events_samples) > 0 or len(self.pause_events_samples) > 0
        return has_fixed or has_spa_events or not self.speech_matrix.empty or not self.pause_matrix.empty
