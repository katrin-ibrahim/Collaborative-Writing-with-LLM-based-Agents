"""
Baselines package for LLM evaluation with streamlined architecture.

This package provides clean implementations for running baseline methods
with improved STORM integration and better error handling.
"""

from .mock_search import MockSearchRM
from .dspy_integration import setup_dspy_for_storm
from .runner import BaselinesRunner

__all__ = [
    'MockSearchRM',
    'setup_dspy_for_storm',
    'BaselinesRunner'
]