"""
Baselines package for LLM evaluation with streamlined architecture.

This package provides clean implementations for running baseline methods
with improved STORM integration and better error handling.
"""

from .llm_wrapper import OllamaLiteLLMWrapper
from .runner import BaselineRunner
from .wikipedia_rm import WikipediaSearchRM

__all__ = ["BaselineRunner", "OllamaLiteLLMWrapper", "WikipediaSearchRM"]
