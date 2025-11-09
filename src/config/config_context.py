# src/utils/config_context.py

import logging
from typing import Optional

from src.collaborative.memory.memory import SharedMemory
from src.config.collaboration_config import CollaborationConfig
from src.config.model_config import ModelConfig
from src.config.retrieval_config import RetrievalConfig
from src.config.storm_config import StormConfig
from src.engines import OllamaEngine, SlurmEngine

logger = logging.getLogger(__name__)


class ConfigContext:
    """
    Singleton configuration context that provides global access to configs and clients.

    Usage:
    1. Initialize once in main: ConfigContext.initialize(configs...)
    2. Use anywhere: ConfigContext.get_client("writing")
    """

    # Singleton state
    _initialized = False
    _model_config: ModelConfig
    _retrieval_config: RetrievalConfig
    _collaboration_config: CollaborationConfig
    _storm_config: StormConfig
    _memory_instance: SharedMemory
    _output_dir: Optional[str] = None

    @classmethod
    def initialize(
        cls,
        model_config: ModelConfig,
        retrieval_config: RetrievalConfig,
        collaboration_config: CollaborationConfig,
        backend: str,
        storm_config: Optional[StormConfig] = None,
        **backend_kwargs,
    ):
        """
        Initialize context once from CLI/main entry point.

        Args:
            model_config: ModelConfig instance
            retrieval_config: RetrievalConfig instance
            collaboration_config: CollaborationConfig instance
            backend: Backend type (ollama, local, etc.)
            storm_config: StormConfig instance (optional)
            backend_kwargs: Additional backend parameters
        """
        if (
            model_config is None
            or retrieval_config is None
            or collaboration_config is None
        ):
            raise RuntimeError(
                "ConfigContext: All configs must be provided and non-None."
            )
        cls._model_config = model_config
        cls._retrieval_config = retrieval_config
        cls._collaboration_config = collaboration_config
        cls._backend = backend
        cls._backend_kwargs = backend_kwargs
        cls._initialized = True

        # Initialize STORM config with defaults if not provided
        if storm_config is None:
            from src.config.storm_config import StormConfig

            storm_config = StormConfig()
            storm_config = storm_config.adapt_to_retrieval_config(retrieval_config)
        cls._storm_config = storm_config
        cls._client_cache = {}

        logger.info("ConfigContext initialized with runner and configs")

    @classmethod
    def get_client(cls, task: str):
        if not cls._initialized or cls._model_config is None:
            raise RuntimeError(
                "ConfigContext is not initialized. Call ConfigContext.initialize(...) before using get_client()."
            )
        if task in cls._client_cache:
            return cls._client_cache[task]

        # Get task-specific config
        model = cls._model_config.get_model_for_task(task)
        temp = cls._model_config.get_temperature_for_task(task)
        max_tokens = cls._model_config.get_token_limit_for_task(task)
        max_tokens = cls._model_config.get_token_limit_for_task(task)

        # Create backend-specific client
        if cls._backend == "ollama":
            client = OllamaEngine(
                host=cls._backend_kwargs.get("ollama_host", "http://localhost:11434"),
                model=model,
                temperature=temp,
                max_tokens=max_tokens,
            )
        elif cls._backend == "slurm":
            client = SlurmEngine(
                device=cls._backend_kwargs.get("device", "cpu"),
                model_path=model,
                temperature=temp,
            )
        else:
            raise ValueError(f"Unsupported backend: {cls._backend}")

        cls._client_cache[task] = client
        return client

    @classmethod
    def get_model_config(cls):
        """Get model configuration."""
        return cls._model_config

    @classmethod
    def get_retrieval_config(cls) -> "RetrievalConfig":
        """Get retrieval configuration."""
        if not cls._initialized or cls._retrieval_config is None:
            raise RuntimeError("ConfigContext: RetrievalConfig not initialized.")
        return cls._retrieval_config

    @classmethod
    def get_collaboration_config(cls) -> "CollaborationConfig":
        """Get collaboration configuration."""
        if not cls._initialized or cls._collaboration_config is None:
            raise RuntimeError("ConfigContext: CollaborationConfig not initialized.")
        return cls._collaboration_config

    @classmethod
    def get_storm_config(cls) -> "StormConfig":
        """Get STORM configuration."""
        if not cls._initialized or cls._storm_config is None:
            raise RuntimeError("ConfigContext: StormConfig not initialized.")
        return cls._storm_config

    @classmethod
    def get_backend(cls) -> str:
        """Get backend type."""
        return cls._backend

    @classmethod
    def get_backend_kwargs(cls) -> dict:
        """Get backend parameters."""
        return cls._backend_kwargs

    @classmethod
    def set_memory_instance(cls, memory):
        """Set the global memory instance for tools and agents."""
        cls._memory_instance = memory
        logger.info("Memory instance registered with ConfigContext")

    @classmethod
    def get_memory_instance(cls) -> "SharedMemory":
        """Get the current memory instance."""
        if cls._memory_instance is None:
            raise RuntimeError("ConfigContext: SharedMemory instance not set.")
        return cls._memory_instance

    @classmethod
    def set_output_dir(cls, output_dir: str):
        """Set the experiment output directory."""
        cls._output_dir = output_dir
        logger.info(f"Output directory registered with ConfigContext: {output_dir}")

    @classmethod
    def get_output_dir(cls) -> Optional[str]:
        """Get the experiment output directory."""
        return cls._output_dir
