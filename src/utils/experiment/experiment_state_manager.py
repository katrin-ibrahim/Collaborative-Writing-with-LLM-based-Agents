from datetime import datetime
from pathlib import Path

import json
import logging
from typing import List

logger = logging.getLogger(__name__)


class ExperimentStateManager:
    """
    Manages experiment state, checkpoints, and resume functionality.

    Handles detection of completed/in-progress/not-started topics per method,
    cleanup of incomplete intermediate files, and progress tracking.
    """

    def __init__(self, output_dir: Path, methods: List[str]):
        self.output_dir = Path(output_dir)
        self.methods = methods
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.results_file = self.output_dir / "results.json"

        logger.info(f"Output directory: {self.output_dir.resolve()}")
        # State tracking
        self.completed_topics = {method: set() for method in methods}
        self.in_progress_topics = {method: set() for method in methods}

    def load_checkpoint(self) -> bool:
        """
        Load existing checkpoint if available.

        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        if not self.checkpoint_file.exists():
            logger.info("No existing checkpoint found, starting fresh")
            return False

        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            for method in self.methods:
                if method in checkpoint_data.get("completed", {}):
                    self.completed_topics[method] = set(
                        checkpoint_data["completed"][method]
                    )
                if method in checkpoint_data.get("in_progress", {}):
                    self.in_progress_topics[method] = set(
                        checkpoint_data["in_progress"][method]
                    )

            logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            self._log_checkpoint_status()
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh experiment")
            return False

    def save_checkpoint(self):
        """Save current state to checkpoint file."""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "completed": {
                method: list(topics) for method, topics in self.completed_topics.items()
            },
            "in_progress": {
                method: list(topics)
                for method, topics in self.in_progress_topics.items()
            },
            "methods": self.methods,
        }

        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def validate_topic_state(self, topic: str, method: str) -> str:
        """
        Validate the completion state of a topic for a given method.
        Checks if article file exists with substantial content (>50 chars).

        Returns:
            'completed', 'in_progress', or 'not_started'
        """
        articles_dir = self.output_dir / "articles"
        safe_topic = topic.replace(" ", "_").replace("/", "_")
        article_file = articles_dir / f"{method}_{safe_topic}.md"
        metadata_file = articles_dir / f"{method}_{safe_topic}_metadata.json"

        if not article_file.exists():
            return "not_started"

        try:
            with open(article_file, "r", encoding="utf-8") as f:
                content = f.read()

            if content and len(content.strip()) > 50:
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        if metadata.get("method") == method:
                            return "completed"
                    except Exception:
                        pass
                return "completed"
            else:
                logger.warning(
                    f"{method} article for {topic} exists but appears incomplete"
                )
                return "in_progress"

        except Exception as e:
            logger.warning(f"Failed to validate {method} article for {topic}: {e}")
            return "in_progress"

    def cleanup_in_progress_topic(self, topic: str, method: str):
        """Clean up incomplete article files for a topic/method."""
        try:
            articles_dir = self.output_dir / "articles"
            safe_topic = topic.replace(" ", "_").replace("/", "_")
            article_file = articles_dir / f"{method}_{safe_topic}.md"
            metadata_file = articles_dir / f"{method}_{safe_topic}_metadata.json"

            if article_file.exists():
                article_file.unlink()
                logger.debug(f"Removed incomplete article: {article_file}")

            if metadata_file.exists():
                metadata_file.unlink()
                logger.debug(f"Removed incomplete metadata: {metadata_file}")

            logger.info(f"Cleaned up incomplete {method} files for topic: {topic}")

        except Exception as e:
            logger.error(f"Failed to cleanup {method} files for {topic}: {e}")

    def mark_topic_completed(self, topic: str, method: str):
        """Mark a topic as completed for a method and save checkpoint."""
        self.completed_topics[method].add(topic)
        self.in_progress_topics[method].discard(topic)
        self.save_checkpoint()
        logger.debug(f"Marked {topic} as completed for {method}")

    def mark_topic_in_progress(self, topic: str, method: str):
        """Mark a topic as in-progress for a specific method."""
        if topic not in self.in_progress_topics[method]:
            self.in_progress_topics[method].add(topic)
            self.save_checkpoint()

    def is_complete(self, topic: str, method: str) -> bool:
        """Check if a topic is completed for a specific method."""
        return topic in self.completed_topics.get(method, set())

    def _log_checkpoint_status(self):
        """Log current checkpoint status for debugging."""
        for method in self.methods:
            completed_count = len(self.completed_topics[method])
            in_progress_count = len(self.in_progress_topics[method])

            if completed_count > 0 or in_progress_count > 0:
                logger.info(
                    f"{method}: {completed_count} completed, {in_progress_count} in progress"
                )
