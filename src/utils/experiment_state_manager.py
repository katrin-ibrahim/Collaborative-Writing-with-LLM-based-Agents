import shutil
from datetime import datetime
from pathlib import Path

import json
import logging
from typing import Dict, List, Optional

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

        Returns:
            'completed', 'in_progress', or 'not_started'
        """
        if method == "storm":
            return self._validate_storm_state(topic)
        elif method == "direct":
            return self._validate_direct_state(topic)
        elif method == "rag":
            return self._validate_rag_state(topic)
        else:
            logger.warning(f"Unknown method {method}, treating as not_started")
            return "not_started"

    def _validate_storm_state(self, topic: str) -> str:
        """Validate STORM completion state for a topic."""
        articles_dir = self.output_dir / "articles"
        storm_outputs_dir = self.output_dir / "storm_outputs"

        # Check for completed article
        article_file = articles_dir / f"{topic}_storm.json"
        if article_file.exists():
            try:
                with open(article_file, "r") as f:
                    article_data = json.load(f)

                # Validate article has required fields and content
                if (
                    article_data.get("content")
                    and len(article_data["content"].strip()) > 100
                    and article_data.get("metadata", {}).get("method") == "storm"
                ):
                    return "completed"
                else:
                    logger.warning(
                        f"STORM article for {topic} exists but appears incomplete"
                    )
                    return "in_progress"

            except Exception as e:
                logger.warning(f"Failed to validate STORM article for {topic}: {e}")
                return "in_progress"

        # Check for intermediate STORM files
        topic_storm_dir = storm_outputs_dir / topic
        if topic_storm_dir.exists() and any(topic_storm_dir.iterdir()):
            return "in_progress"

        return "not_started"

    def _validate_direct_state(self, topic: str) -> str:
        """Validate direct prompting completion state for a topic."""
        articles_dir = self.output_dir / "articles"
        article_file = articles_dir / f"{topic}_direct.json"

        if article_file.exists():
            try:
                with open(article_file, "r") as f:
                    article_data = json.load(f)

                # Validate article has required fields and content
                if (
                    article_data.get("content")
                    and len(article_data["content"].strip()) > 50
                    and article_data.get("metadata", {}).get("method") == "direct"
                ):
                    return "completed"
                else:
                    logger.warning(
                        f"Direct article for {topic} exists but appears incomplete"
                    )
                    return "in_progress"

            except Exception as e:
                logger.warning(f"Failed to validate direct article for {topic}: {e}")
                return "in_progress"

        return "not_started"

    def _validate_rag_state(self, topic: str) -> str:
        """Validate RAG completion state for a topic."""
        articles_dir = self.output_dir / "articles"
        article_file = articles_dir / f"{topic}_rag.json"

        if article_file.exists():
            try:
                with open(article_file, "r") as f:
                    article_data = json.load(f)

                # Validate article has required fields and content
                if (
                    article_data.get("content")
                    and len(article_data["content"].strip()) > 50
                    and article_data.get("metadata", {}).get("method") == "rag"
                ):
                    return "completed"
                else:
                    logger.warning(
                        f"RAG article for {topic} exists but appears incomplete"
                    )
                    return "in_progress"

            except Exception as e:
                logger.warning(f"Failed to validate RAG article for {topic}: {e}")
                return "in_progress"

        return "not_started"

    def cleanup_in_progress_topic(self, topic: str, method: str):
        """Clean up incomplete intermediate files for a topic/method."""
        try:
            if method == "storm":
                self._cleanup_storm_intermediate(topic)
            elif method == "direct":
                self._cleanup_direct_intermediate(topic)
            elif method == "rag":
                self._cleanup_rag_intermediate(topic)

            logger.info(f"Cleaned up incomplete {method} files for topic: {topic}")

        except Exception as e:
            logger.error(f"Failed to cleanup {method} files for {topic}: {e}")

    def _cleanup_storm_intermediate(self, topic: str):
        """Clean up incomplete STORM intermediate files."""
        storm_outputs_dir = self.output_dir / "storm_outputs"
        topic_storm_dir = storm_outputs_dir / topic

        if topic_storm_dir.exists():
            shutil.rmtree(topic_storm_dir)
            logger.debug(f"Removed STORM intermediate directory: {topic_storm_dir}")

        # Also remove incomplete article file
        articles_dir = self.output_dir / "articles"
        article_file = articles_dir / f"{topic}_storm.json"
        if article_file.exists():
            article_file.unlink()
            logger.debug(f"Removed incomplete STORM article: {article_file}")

    def _cleanup_direct_intermediate(self, topic: str):
        """Clean up incomplete direct prompting files."""
        articles_dir = self.output_dir / "articles"
        article_file = articles_dir / f"{topic}_direct.json"

        if article_file.exists():
            article_file.unlink()
            logger.debug(f"Removed incomplete direct article: {article_file}")

    def _cleanup_rag_intermediate(self, topic: str):
        """Clean up incomplete RAG files."""
        articles_dir = self.output_dir / "articles"
        article_file = articles_dir / f"{topic}_rag.json"

        if article_file.exists():
            article_file.unlink()
            logger.debug(f"Removed incomplete RAG article: {article_file}")

    def analyze_existing_state(self, topics: List[str]) -> Dict[str, Dict[str, str]]:
        """
        Analyze the current state of all topics for all methods.

        Returns:
            Dict mapping topic -> method -> state ('completed'/'in_progress'/'not_started')
        """
        topic_states = {}

        for topic in topics:
            topic_states[topic] = {}
            for method in self.methods:
                state = self.validate_topic_state(topic, method)
                topic_states[topic][method] = state

                # Update internal tracking
                if state == "completed":
                    self.completed_topics[method].add(topic)
                elif state == "in_progress":
                    self.in_progress_topics[method].add(topic)

        return topic_states

    def get_remaining_topics(self, all_topics: List[str]) -> Dict[str, List[str]]:
        """
        Get topics that still need to be processed for each method.

        Returns:
            Dict mapping method -> list of remaining topics
        """
        remaining = {}

        for method in self.methods:
            remaining[method] = [
                topic
                for topic in all_topics
                if topic not in self.completed_topics[method]
            ]

        return remaining

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

    def cleanup_and_restart_in_progress(self, topics: List[str]):
        """Clean up all in-progress topics and prepare for restart."""
        cleanup_count = 0

        for topic in topics:
            for method in self.methods:
                if topic in self.in_progress_topics[method]:
                    self.cleanup_in_progress_topic(topic, method)
                    self.in_progress_topics[method].discard(topic)
                    cleanup_count += 1

        if cleanup_count > 0:
            logger.info(
                f"Cleaned up {cleanup_count} in-progress topic/method combinations"
            )
            self.save_checkpoint()

    def _log_checkpoint_status(self):
        """Log current checkpoint status for debugging."""
        for method in self.methods:
            completed_count = len(self.completed_topics[method])
            in_progress_count = len(self.in_progress_topics[method])

            if completed_count > 0 or in_progress_count > 0:
                logger.info(
                    f"{method}: {completed_count} completed, {in_progress_count} in progress"
                )

    def get_progress_summary(self, total_topics: int) -> Dict[str, Dict[str, int]]:
        """Get progress summary for all methods."""
        summary = {}

        for method in self.methods:
            completed = len(self.completed_topics[method])
            remaining = total_topics - completed

            summary[method] = {
                "completed": completed,
                "remaining": remaining,
                "total": total_topics,
                "progress_pct": (
                    round((completed / total_topics) * 100, 1)
                    if total_topics > 0
                    else 0
                ),
            }

        return summary

    def load_existing_results(self) -> Dict:
        """Load existing results file if available."""
        if not self.results_file.exists():
            return {}

        try:
            with open(self.results_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load existing results: {e}")
            return {}

    def find_latest_run_dir(self, base_output_dir: Path) -> Optional[Path]:
        """Find the most recent run directory for resuming."""
        if not base_output_dir.exists():
            return None

        run_dirs = [
            d
            for d in base_output_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

        if not run_dirs:
            return None

        # Sort by modification time and return most recent
        latest_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
        return latest_dir
