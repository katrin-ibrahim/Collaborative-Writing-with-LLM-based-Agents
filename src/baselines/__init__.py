"""
Baselines package for LLM evaluation with unified architecture.

This package provides implementations for running baseline methods
with both Ollama and local backends using a consistent interface.
"""

# Re-export key components
from .cli_args import parse_arguments
from .runner_factory import create_runner

__all__ = ["parse_arguments", "create_runner"]
