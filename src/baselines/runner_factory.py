"""
Factory for creating the appropriate runner implementation.
"""

from typing import Tuple, Type

from src.baselines.local_baselines import LocalBaselineRunner
from src.baselines.ollama_baselines import BaselineRunner as OllamaRunner


def create_runner(backend: str) -> Tuple[Type, str]:
    """
    Create the appropriate runner class for the specified backend.

    Args:
        backend: The backend type ('ollama' or 'local')

    Returns:
        Tuple of (RunnerClass, runner_name)

    Raises:
        ValueError: If an invalid backend is specified
    """
    if backend == "ollama":
        return OllamaRunner, "Ollama"
    elif backend == "local":
        return LocalBaselineRunner, "Local"
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'ollama' or 'local'.")
