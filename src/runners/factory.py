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
        from src.runners.ollama_runner import OllamaRunner

        return OllamaRunner, "Ollama"
    elif backend == "slurm":
        from src.runners.slurm_runner import SlurmRunner

        return SlurmRunner, "SLURM"
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'ollama' or 'slurm'.")
