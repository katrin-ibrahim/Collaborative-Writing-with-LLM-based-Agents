"""
Factory for creating the appropriate runner implementation.
"""

from typing import Tuple, Type


def create_runner(backend: str) -> Tuple[Type, str]:
    """
    Create the appropriate runner class for the specified backend.

    Args:
        backend: The backend type ('ollama' or 'slurm')

    Returns:
        Tuple of (RunnerClass, runner_name)

    Raises:
        ValueError: If an invalid backend is specified
    """
    if backend == "ollama":
        from src.baselines.ollama_baselines.ollama_runner import (
            BaselineRunner as OllamaRunner,
        )

        return OllamaRunner, "Ollama"
    elif backend == "slurm":
        from src.baselines.slurm_baselines.slurm_runner import SlurmBaselineRunner

        return SlurmBaselineRunner, "SLURM"
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'ollama' or 'slurm'.")
