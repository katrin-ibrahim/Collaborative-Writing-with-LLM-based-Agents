import sys
from pathlib import Path

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# add src
sys.path.append(str(Path(__file__).resolve().parent))


@dataclass
class FreshWikiEntry:
    """
    Simple FreshWiki evaluation entry.
    All entries are pre-filtered for quality by the extraction script.
    """

    topic: str
    reference_outline: List[str]
    reference_content: str
    metadata: dict


class FreshWikiLoader:
    """
    Simple loader for pre-filtered quality FreshWiki dataset.
    """

    def __init__(self, data_path: str = "data/freshwiki"):
        self.data_path = Path(data_path)
        logger.info(f"Loading FreshWiki dataset from {self.data_path.resolve()}")
        self.entries: List[FreshWikiEntry] = []

        # Load all pre-filtered entries
        self._load_entries()

    def _load_entries(self) -> None:
        """Load all quality-filtered FreshWiki entries."""

        if not self.data_path.exists():
            logger.error(f"FreshWiki data not found at: {self.data_path}")
            logger.error(
                "Run: python src/utils/extract_quality_freshwiki.py /path/to/FreshWiki"
            )
            return

        json_dir = self.data_path / "json"
        txt_dir = self.data_path / "txt"

        if not json_dir.exists() or not txt_dir.exists():
            logger.error(
                "FreshWiki subdirectories not found. Run extraction script first."
            )
            return

        # Find matching JSON and TXT files
        json_files = list(json_dir.glob("*.json"))

        logger.info(f"Loading {len(json_files)} quality-filtered FreshWiki entries...")

        # Load all entries (no validation needed - already quality filtered)
        for json_file in json_files:
            txt_file = txt_dir / f"{json_file.stem}.txt"

            if not txt_file.exists():
                logger.warning(f"Missing text file for {json_file.name}")
                continue

            try:
                # Load JSON metadata
                with open(json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)

                # Load text content
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Create entry
                entry = FreshWikiEntry(
                    topic=json_data.get("title", json_file.stem.replace("_", " ")),
                    reference_outline=json_data.get("sections", []),
                    reference_content=content,
                    metadata={
                        "url": json_data.get("url", ""),
                        "summary": json_data.get("summary", ""),
                        "word_count": json_data.get("word_count", len(content.split())),
                        "source_file": json_data.get("source_file", ""),
                        "quality_filtered": True,
                    },
                )

                self.entries.append(entry)

            except Exception as e:
                logger.warning(f"Failed to load {json_file.name}: {e}")

        logger.info(
            f"Successfully loaded {len(self.entries)} quality FreshWiki entries"
        )

    def get_evaluation_sample(self, n: int = 5) -> List[FreshWikiEntry]:
        """Get first n entries in deterministic order for reproducible experiments."""
        if not self.entries:
            logger.error("No FreshWiki entries available. Run extraction script first.")
            return []

        # Simply take the first n entries (they're already loaded in consistent order)
        sample_size = min(n, len(self.entries))
        if sample_size < n:
            logger.warning(f"Requested {n} topics but only {sample_size} available")

        selected = self.entries[:sample_size]

        return selected

    def get_entry_by_topic(self, topic: str) -> Optional[FreshWikiEntry]:
        """Get specific entry by topic name."""
        for entry in self.entries:
            if entry.topic.lower() == topic.lower():
                return entry
        return None

    def get_all_topics(self) -> List[str]:
        """Get list of all available topics."""
        return [entry.topic for entry in self.entries]

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        if not self.entries:
            return {
                "status": "no_data",
                "message": "No entries loaded. Run extraction script first.",
                "total_entries": 0,
            }

        word_counts = [len(entry.reference_content.split()) for entry in self.entries]
        section_counts = [len(entry.reference_outline) for entry in self.entries]

        return {
            "status": "loaded",
            "total_entries": len(self.entries),
            "avg_word_count": round(sum(word_counts) / len(word_counts), 1),
            "avg_sections": round(sum(section_counts) / len(section_counts), 1),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "min_sections": min(section_counts),
            "max_sections": max(section_counts),
            "quality_filtered": True,
            "sample_topics": [entry.topic for entry in self.entries[:5]],
        }
