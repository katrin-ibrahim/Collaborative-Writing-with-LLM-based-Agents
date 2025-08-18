import shutil
from datetime import datetime
from pathlib import Path

import json
import logging

from src.utils.data import Article

logger = logging.getLogger(__name__)


class OutputManager:
    """Unified output management for all baseline methods."""

    @staticmethod
    def create_output_dir(
        backend: str,
        methods: list,
        num_topics: int,
        timestamp: str = None,
        custom_name: str = None,
    ) -> str:
        """
        Create standardized output directory path following the convention:
        results/[backend]/[method(s)]_N=[num_of_topics]_T=[timestamp]

        If custom_name is provided, it replaces the auto-generated name part:
        results/[backend]/[custom_name]

        Args:
            backend: The backend name (e.g., "ollama", "slurm")
            methods: List of methods to be used
            num_topics: Number of topics to process
            timestamp: Optional timestamp, if not provided current time will be used
            custom_name: Optional custom name to override the auto-generated name

        Returns:
            Full directory path as string
        """
        # If custom name is provided, use it instead of auto-generated name
        if custom_name:
            return f"results/{backend}/{custom_name}"

        # Use current timestamp if not provided
        if timestamp is None:
            now = datetime.now()
            timestamp = now.strftime("%d.%m_%H:%M")

        # Determine method name for directory
        if set(methods) == {"direct", "storm", "rag"}:
            method_str = "all"
        else:
            method_str = "_".join(methods)

        # Create output directory with correct structure
        return f"results/{backend}/{method_str}_N={num_topics}_T={timestamp}"

    @staticmethod
    def verify_resume_dir(resume_dir: str) -> str:
        """
        Verify a resume directory exists and is valid.

        Args:
            resume_dir: Path to the directory to resume from

        Returns:
            The verified directory path

        Raises:
            ValueError: If directory doesn't exist
        """
        dir_path = Path(resume_dir)
        if not dir_path.exists():
            raise ValueError(f"Resume directory does not exist: {resume_dir}")

        # Verify this is likely a valid experiment directory
        if not (dir_path / "articles").exists():
            logger.warning(
                f"Resume directory may not be a valid experiment directory: {resume_dir}"
            )

        return str(dir_path)

    def __init__(self, base_output_dir: str, debug_mode: bool = False):
        self.base_dir = Path(base_output_dir)
        self.debug_mode = debug_mode

        # Create directory structure
        self.articles_dir = self.base_dir / "articles"
        logger.info(f"Base output directory: {self.base_dir.resolve()}")
        self.articles_dir.mkdir(parents=True, exist_ok=True)

        if self.debug_mode:
            self.debug_dir = self.base_dir / "debug"
            logger.info(f"Debug output directory: {self.debug_dir.resolve()}")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Debug mode enabled - intermediate files will be saved")

    def save_article(self, article: Article, method: str) -> Path:
        """Save article content and metadata to standardized location."""
        base_filename = f"{method}_{article.title.replace(' ', '_').replace('/', '_')}"
        content_filepath = self.articles_dir / f"{base_filename}.md"
        metadata_filepath = self.articles_dir / f"{base_filename}_metadata.json"

        try:
            # Save article content
            with open(content_filepath, "w", encoding="utf-8") as f:
                f.write(article.content)

            # Save article metadata including generation time and word count
            metadata = {
                "title": article.title,
                "method": method,
                "generation_time": article.metadata.get("generation_time", 0.0),
                "word_count": article.metadata.get(
                    "word_count", len(article.content.split())
                ),
                "model": article.metadata.get("model", "unknown"),
                "timestamp": article.metadata.get("timestamp", "unknown"),
                **article.metadata,  # Include all other metadata
            }
            additional_metadata = article._serialize_metadata(article.metadata)
            for key, value in additional_metadata.items():
                if key not in metadata:  # Don't overwrite existing clean values
                    metadata[key] = value

            with open(metadata_filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved {method} article and metadata: {content_filepath}")
            return content_filepath
        except Exception as e:
            logger.error(f"Failed to save article {content_filepath}: {e}")
            raise

    def setup_storm_output_dir(self, topic) -> str:
        """
        Setup STORM output directory based on debug mode.

        Args:
            topic: Topic string or FreshWikiEntry object

        Returns:
            String path to the output directory
        """
        # Extract topic string if it's a FreshWikiEntry object
        topic_str = topic.topic if hasattr(topic, "topic") else str(topic)

        if self.debug_mode:
            # Use debug directory for STORM intermediate files
            storm_dir = (
                self.debug_dir / "storm" / topic_str.replace(" ", "_").replace("/", "_")
            )
            storm_dir.mkdir(parents=True, exist_ok=True)
            return str(storm_dir)
        else:
            # Use temporary directory that gets cleaned up
            temp_dir = (
                self.base_dir
                / "temp_storm"
                / topic_str.replace(" ", "_").replace("/", "_")
            )
            temp_dir.mkdir(parents=True, exist_ok=True)
            return str(temp_dir)

    def cleanup_storm_temp(self, topic):
        """
        Clean up temporary STORM files if not in debug mode.

        Args:
            topic: Topic string or FreshWikiEntry object
        """
        # Extract topic string if it's a FreshWikiEntry object
        topic_str = topic.topic if hasattr(topic, "topic") else str(topic)

        if not self.debug_mode:
            temp_dir = (
                self.base_dir
                / "temp_storm"
                / topic_str.replace(" ", "_").replace("/", "_")
            )
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug(f"Cleaned up temporary STORM files for {topic_str}")
                logger.debug(f"Cleaned up temporary STORM files for {topic}")
