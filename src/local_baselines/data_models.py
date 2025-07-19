"""
Data models for local experiments.
Uses shared data models from utils for compatibility.
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataclasses import dataclass

# Import shared Article model for compatibility with baselines


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    experiment_name: str
    methods: list
    topic_limit: int
    model_path: str
    output_dir: str
    data_source: str
    resume: bool = False
    debug: bool = False
