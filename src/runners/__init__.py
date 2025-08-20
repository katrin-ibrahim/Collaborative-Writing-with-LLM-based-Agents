"""
Runners package for collaborative writing experiments.

This package provides factory-based runners that use the method factory pattern
for clean separation between infrastructure and method implementations.
"""

from src.runners.base_runner import BaseRunner
from src.runners.factory import create_runner
from src.runners.ollama_runner import OllamaRunner
from src.runners.slurm_runner import SlurmRunner

from .cli_args import parse_arguments

__all__ = [
    "BaseRunner",
    "OllamaRunner",
    "SlurmRunner",
    "create_runner",
    "parse_arguments",
]
