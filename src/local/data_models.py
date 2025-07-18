"""
Data models for local experiments.
Standalone version without external dependencies.
"""
from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class Article:
    """Data model for generated articles."""
    title: str
    content: str
    word_count: int = 0
    generation_time: float = 0.0
    method: str = "direct_prompting"
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.content.split()) if self.content else 0
        if self.timestamp == 0.0:
            self.timestamp = time.time()


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
