"""
Baselines package for LLM evaluation with streamlined architecture.

This package provides clean implementations for running baseline methods
with improved STORM integration and better error handling.
"""

from .runner import BaselineRunner
from .llm_wrapper import OllamaLiteLLMWrapper
from .wikipedia_search import WikipediaSearchRM


__all__ = [
    "BaselineRunner",
    "OllamaLiteLLMWrapper",
    "WikipediaSearchRM"
]