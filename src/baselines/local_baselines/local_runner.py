"""
Ultra-thin local baseline runner - only responsible for engine initialization.
All shared logic moved to BaseRunner.
"""

import logging
from typing import Optional

from baselines.main_runner_base import BaseRunner
from src.baselines.model_engines.local_engine import LocalModelEngine
from src.config.baselines_model_config import ModelConfig
from src.utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class LocalBaselineRunner(BaseRunner):
    """
    Ultra-thin local runner - only handles LocalModelEngine initialization.
    All methods (direct, rag, batch) implemented in BaseRunner.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        output_manager: Optional[OutputManager] = None,
        device: str = "auto",
        model_path: Optional[str] = None,
    ):
        # Initialize base class
        super().__init__(model_config=model_config, output_manager=output_manager)

        # Ensure local mode
        self.model_config.mode = "local"

        # Store local-specific parameters
        self.device = device
        self.model_path = model_path

        # Engine cache
        self._engine_cache = {}

        logger.info("LocalBaselineRunner initialized")
        logger.info(f"Using device: {self.device}")

    def get_model_engine(self, task: str) -> LocalModelEngine:
        """Get cached LocalModelEngine for the specified task."""
        if task in self._engine_cache:
            return self._engine_cache[task]

        # Create new engine
        model_path = self.model_path or self.model_config.get_model_path(task)

        engine = LocalModelEngine(
            model_path=model_path,
            device=self.device,
            config=self.model_config,
            task=task,
        )

        # Cache and return
        self._engine_cache[task] = engine
        return engine

    def get_supported_methods(self):
        """Return methods supported by local runner."""
        return ["direct", "rag"]  # No STORM for local

    # All other methods (run_direct, run_rag, run_*_batch) implemented in BaseRunner
