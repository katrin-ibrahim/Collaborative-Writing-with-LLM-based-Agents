"""
SLURM runner using factory pattern for collaborative methods.
Handles local model engine initialization, inherits orchestration from BaseRunner.
"""

import logging
from typing import Optional

from src.config.baselines_model_config import ModelConfig
from src.config.collaboration_config import CollaborationConfig
from src.config.retrieval_config import RetrievalConfig
from src.runners.base_runner import BaseRunner
from src.utils.io import OutputManager

logger = logging.getLogger(__name__)


class SlurmRunner(BaseRunner):
    """
    SLURM runner for collaborative methods.
    Handles local model engine initialization, uses factory for all methods.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        collaboration_config: Optional[CollaborationConfig] = None,
        device: str = "auto",
        model_path: Optional[str] = None,
    ):
        # Initialize base class
        super().__init__(
            model_config=model_config,
            output_manager=output_manager,
            retrieval_config=retrieval_config,
            collaboration_config=collaboration_config,
        )

        # Ensure local mode
        self.model_config.mode = "local"

        # Store local-specific parameters
        self.device = device
        self.model_path = model_path

        # Import SLURM-specific components
        try:
            from src.baselines.slurm_baselines.slurm_engine import LocalModelEngine

            self.LocalModelEngine = LocalModelEngine
        except ImportError as e:
            logger.error(f"Failed to import SLURM dependencies: {e}")
            raise RuntimeError(
                "SLURM backend requires local model dependencies. "
                "Please ensure SLURM baseline components are available."
            )

        # Engine cache for efficiency
        self._engine_cache = {}

        logger.info("SlurmRunner initialized:")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Model mode: {self.model_config.mode}")
        logger.info(
            f"  - Collaboration max_iterations: {self.collaboration_config.max_iterations}"
        )
        logger.info(f"  - Supported methods: {self.get_supported_methods()}")

    def get_client(self):
        """Get the local model engine for method initialization."""
        # For SLURM, we use the model engine as the "client"
        # Use cached engine for the default task or create new one
        task = "writing"  # Default task for collaborative methods

        if task in self._engine_cache:
            return self._engine_cache[task]

        # Create new engine
        model_path = self.model_path or self.model_config.get_model_path(task)

        engine = self.LocalModelEngine(
            model_path=model_path,
            device=self.device,
            config=self.model_config,
            task=task,
        )

        # Cache and return
        self._engine_cache[task] = engine
        return engine
