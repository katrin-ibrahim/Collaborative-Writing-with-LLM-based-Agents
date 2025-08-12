"""
Ultra-thin local baseline runner - only responsible for engine initialization.
All shared logic moved to BaseRunner.
"""

import logging
from typing import List, Optional

from src.baselines.baseline_runner_base import BaseRunner
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

    def _get_query_generator(self):
        """Local implementation of query generator."""

        def local_query_generator(
            engine, topic: str, num_queries: int = 5
        ) -> List[str]:
            prompt = f"""Generate {num_queries} specific Wikipedia search queries about "{topic}".

    IMPORTANT: Make each query specific and contextual to avoid ambiguous results.
    Include key context words from the topic to ensure relevant results.

    Examples:
    - Topic: "Music in major and minor keys"
      Good: "musical key signatures major minor", "major minor scales music theory"
      Bad: "key signature" (could return cryptography), "scales" (too generic)

    - Topic: "Digital photography techniques"
      Good: "digital camera photography techniques", "photo editing digital methods"
      Bad: "exposure" (could return business/medical), "filters" (too generic)

    Generate contextual search queries for "{topic}":
    Each query should include topic-specific context words to ensure relevant Wikipedia results.

    Output ONLY the search queries, one per line, with NO numbering or explanations.

    Wikipedia search queries for "{topic}":"""

            try:
                response = engine.generate(prompt, max_length=200, temperature=0.3)

                # Extract content
                if hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)

                # Parse queries
                raw_queries = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip()
                    and not line.strip().startswith(("1.", "2.", "3.", "-", "*"))
                ]

                # Return queries with fallback
                return raw_queries[:num_queries] if raw_queries else [topic]

            except Exception as e:
                logger.error(f"Local query generation failed: {e}")
                return [topic]  # Fallback to topic

        return local_query_generator

    def get_supported_methods(self):
        """Return methods supported by local runner."""
        return ["direct", "rag", "agentic", "collaborative"]  # No STORM for local

    # All other methods (run_direct, run_rag, run_*_batch) implemented in BaseRunner
