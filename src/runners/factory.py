"""
Factory for creating the appropriate runner implementation.
"""

from src.runners import BaseRunner


def create_runner(
    backend: str,
    model_config=None,
    output_manager=None,
    retrieval_config=None,
    collaboration_config=None,
    **backend_kwargs,
) -> BaseRunner:
    """
    Create and return a fully initialized runner instance.
    """
    if backend == "ollama":
        from src.runners.ollama_runner import OllamaRunner

        return OllamaRunner(
            model_config=model_config,
            output_manager=output_manager,
            retrieval_config=retrieval_config,
            collaboration_config=collaboration_config,
            **backend_kwargs,  # ollama_host, etc.
        )
    elif backend == "slurm":
        from src.runners.slurm_runner import SlurmRunner

        return SlurmRunner(
            model_config=model_config,
            output_manager=output_manager,
            retrieval_config=retrieval_config,
            collaboration_config=collaboration_config,
            **backend_kwargs,  # device, model_path, etc.
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
