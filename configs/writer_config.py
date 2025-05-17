import os
from dataclasses import dataclass, field

@dataclass
class WriterConfig:
    """Configuration for the Writer model"""
    model_name: str = "OpenAssistant/oasst-sft-1-pythia-12b"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2000
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN")) 