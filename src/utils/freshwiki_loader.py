import sys
from pathlib import Path

import json
import logging
from dataclasses import dataclass
from typing import List

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

    def __init__(
        self,
        data_path: str = "/Users/katrin/Documents/Repos/Collaborative-Writing-with-LLM-based-Agents/data/freshwiki",
    ):
        self.data_path = Path(data_path)
        abs_path = self.data_path.resolve()
        logger.info(f"Loading FreshWiki dataset from {abs_path}")

    def load_topics(self, num_topics: int = 5) -> List[FreshWikiEntry]:
        """Load only the requested number of topics efficiently."""
        if not self.data_path.exists():
            logger.error(f"FreshWiki data not found at: {self.data_path}")
            return []

        json_files = list((self.data_path / "json").glob("*.json"))[:num_topics]

        entries = []
        for json_file in json_files:
            # Load only what we need
            txt_file = self.data_path / "txt" / f"{json_file.stem}.txt"

            try:
                with open(json_file) as f:
                    json_data = json.load(f)
                with open(txt_file) as f:
                    content = f.read()

                entries.append(
                    FreshWikiEntry(
                        topic=json_data["title"],
                        reference_outline=json_data["sections"],
                        reference_content=content,
                        metadata=json_data,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return entries
