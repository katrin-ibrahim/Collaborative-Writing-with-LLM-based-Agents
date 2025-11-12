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
        data_path: str = "/Users/katrin/Documents/Repos/Collaborative-Writing-with-LLM-based-Agents/data/fw",
        # data_path: str = "/storage/ukp/work/ibrahim1/Writer-Reviewer/data/freshwiki",
    ):
        self.data_path = Path(data_path)
        abs_path = self.data_path.resolve()
        logger.info(f"Loading FreshWiki dataset from {abs_path}")

    def load_topics(self, num_topics: int = 5) -> List[FreshWikiEntry]:
        """Load only the requested number of topics efficiently."""
        if not self.data_path.exists():
            logger.error(f"FreshWiki data not found at: {self.data_path}")
            return []

        json_files = sorted(list((self.data_path / "json").glob("*.json")))[:num_topics]

        entries = []
        for json_file in json_files:
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

    def load_topics_by_name(self, topic_names: List[str]) -> List[FreshWikiEntry]:
        """Load specific topics by name, only reading necessary files."""
        if not self.data_path.exists():
            logger.error(f"FreshWiki data not found at: {self.data_path}")
            return []

        normalized_names = {
            name.replace("_", " ").lower(): name for name in topic_names
        }
        entries = []

        for json_file in (self.data_path / "json").glob("*.json"):
            try:
                with open(json_file) as f:
                    json_data = json.load(f)

                topic_title = json_data.get("title", "")
                normalized_title = topic_title.replace("_", " ").lower()

                if normalized_title in normalized_names:
                    txt_file = self.data_path / "txt" / f"{json_file.stem}.txt"
                    with open(txt_file) as f:
                        content = f.read()

                    entries.append(
                        FreshWikiEntry(
                            topic=topic_title,
                            reference_outline=json_data["sections"],
                            reference_content=content,
                            metadata=json_data,
                        )
                    )

                    if len(entries) == len(topic_names):
                        break

            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return entries
