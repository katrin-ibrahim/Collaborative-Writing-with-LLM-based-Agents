# src/utils/config_context.py

import logging

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
    _model_config = None
    _retrieval_config = None
    _collaboration_config = None
    _memory_instance = None

    @classmethod
    def initialize(
        cls,
        model_config,
        retrieval_config,
        collaboration_config,
        backend,
        **backend_kwargs
    ):
        """
        Initialize context once from CLI/main entry point.

        Args:
            model_config: ModelConfig instance
            retrieval_config: RetrievalConfig instance
            collaboration_config: CollaborationConfig instance
            backend: Backend type (ollama, local, etc.)
            backend_kwargs: Additional backend parameters
        """
        cls._model_config = model_config
        cls._retrieval_config = retrieval_config
        cls._collaboration_config = collaboration_config
        cls._backend = backend
        cls._backend_kwargs = backend_kwargs
        cls._client_cache = {}

        logger.info("ConfigContext initialized with runner and configs")

    @classmethod
    def get_client(cls, task: str):
        if task in cls._client_cache:
            return cls._client_cache[task]

        # Get task-specific config
        model = cls._model_config.get_model_for_task(task)
        temp = cls._model_config.get_temperature_for_task(task)

        # Create backend-specific client
        if cls._backend == "ollama":
            client = OllamaEngine(
                host=cls._backend_kwargs.get("ollama_host"),
                model=model,
                temperature=temp,
            )
        elif cls._backend == "slurm":
            client = SlurmEngine(
                device=cls._backend_kwargs.get("device"),
                model_path=model,
                temperature=temp,
            )

        cls._client_cache[task] = client
        return client

    @classmethod
    def get_model_config(cls):
        """Get model configuration."""
        return cls._model_config

    @classmethod
    def get_retrieval_config(cls):
        """Get retrieval configuration."""
        return cls._retrieval_config

    @classmethod
    def get_collaboration_config(cls):
        """Get collaboration configuration."""
        return cls._collaboration_config

    @classmethod
    def get_backend(cls):
        """Get backend type."""
        return cls._backend

    @classmethod
    def get_backend_kwargs(cls):
        """Get backend parameters."""
        return cls._backend_kwargs

    @classmethod
    def set_memory_instance(cls, memory):
        """Set the global memory instance for tools and agents."""
        cls._memory_instance = memory
        logger.info("Memory instance registered with ConfigContext")

    @classmethod
    def get_memory_instance(cls):
        """Get the current memory instance."""
        return cls._memory_instance
