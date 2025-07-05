"""
Baselines package for LLM evaluation and testing.

This package provides clean implementations for running baseline methods
like STORM with local models, handling all the integration complexities.
"""

from .mock_search import MockSearchRM
from .llm_wrapper import LocalLiteLLMWrapper
from .dspy_integration import setup_dspy_integration
from .runner import BaselinesRunner

__all__ = [
    'MockSearchRM',
    'LocalLiteLLMWrapper', 
    'setup_dspy_integration',
    'BaselinesRunner'
]