import os
from dataclasses import dataclass

@dataclass
class RunnerArgument:
    """Arguments for controlling the pipeline"""
    topic: str
    corpus_embeddings_path: str  # path to persisted FAISS index
    dataset_name: str = "YuchengJiang/WildSeek"
    retrieve_top_k: int = 5
    research_phase_enabled: bool = True
    outline_generation_enabled: bool = True