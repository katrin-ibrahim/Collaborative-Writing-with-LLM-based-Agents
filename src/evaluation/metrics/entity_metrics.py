# src/evaluation/metrics/entity_metrics.py
import logging
import re
from typing import Set

logger = logging.getLogger(__name__)


class EntityMetrics:
    """
    Entity extraction using simple heuristics.
    Focuses on patterns that work well without heavy NLP dependencies.
    """

    def extract_entities(self, text: str) -> Set[str]:
        """
        Extract entities using simple but effective heuristics.

        Combines multiple simple approaches:
        1. Capitalized word sequences (names, places, organizations)
        2. Number patterns (dates, statistics, money)
        3. Common entity markers
        4. Simple word matching for heading overlap
        """

        logger.debug(f"Extracting entities from text: {text[:200]}...")
        entities = set()

        # 1. Simple word extraction for headings (improved for HER calculation)
        # Extract individual meaningful words (lowercased) for basic overlap
        import re

        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "with",
            "was",
            "this",
            "that",
            "have",
            "from",
            "will",
            "can",
            "not",
        }
        for word in words:
            if word not in stop_words and len(word) >= 3:
                entities.add(word)
                logger.debug(f"Added word entity: {word}")

        # 2. Multi-word phrases for compound headings
        clean_text = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        if " " in clean_text and len(clean_text) > 3:
            entities.add(clean_text)
            logger.debug(f"Added phrase entity: {clean_text}")

        # 3. Capitalized sequences (names, places, organizations)
        # Matches "John Smith", "New York", "Google Inc"
        cap_sequences = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        logger.debug(
            f"Found {len(cap_sequences)} capitalized sequences: {cap_sequences}"
        )
        for seq in cap_sequences:
            # Filter out sentence starters and common words
            if not self._is_sentence_starter(seq, text) and len(seq) > 2:
                entities.add(seq.lower())
                logger.debug(f"Added capitalized entity: {seq.lower()}")
            else:
                logger.debug(f"Filtered out: {seq} (sentence starter or too short)")

        # 4. Numbers with context (years, percentages, money)
        number_patterns = [
            r"\b(19|20)\d{2}\b",  # Years
            r"\b\d+(?:\.\d+)?%\b",  # Percentages
            r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b",  # Money
            r"\b\d+(?:\.\d+)?\s*(?:million|billion|thousand)\b",  # Large numbers
        ]

        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(
                match.lower() if isinstance(match, str) else str(match).lower()
                for match in matches
            )

        # 5. Common entity markers
        # Things like "Dr. Smith", "President Obama"
        title_patterns = [
            r"\b(?:Dr|Prof|President|CEO|Director)\.\s+[A-Z][a-z]+\b",
            r"\b(?:Mr|Ms|Mrs)\.\s+[A-Z][a-z]+\b",
        ]

        for pattern in title_patterns:
            matches = re.findall(pattern, text)
            if matches:
                logger.debug(f"Title pattern found {len(matches)} matches: {matches}")
            entities.update(match.lower() for match in matches)

        logger.debug(f"Total entities extracted: {len(entities)} - {entities}")
        return entities

    def _is_sentence_starter(self, word_sequence: str, full_text: str) -> bool:
        """Check if capitalized sequence is just a sentence starter."""
        # Simple heuristic: if it appears after '. ' or at start, likely sentence starter
        sentence_starters = {
            "The",
            "This",
            "That",
            "These",
            "Those",
            "A",
            "An",
            "In",
            "On",
            "At",
            "For",
        }
        first_word = word_sequence.split()[0]

        # Check if it's a common sentence starter
        if first_word in sentence_starters:
            return True

        # Check if it appears after periods (sentence boundaries)
        pattern = r"\.\s+" + re.escape(word_sequence)
        return bool(re.search(pattern, full_text))

    def calculate_overall_entity_recall(self, generated: str, reference: str) -> float:
        """
        Calculate Article Entity Recall (AER) using simple entity extraction.

        This gives a good approximation of factual content overlap
        without complex NLP dependencies.
        """
        logger.debug("=== Article Entity Recall Calculation ===")
        gen_entities = self.extract_entities(generated)
        ref_entities = self.extract_entities(reference)

        logger.debug(f"Generated entities ({len(gen_entities)}): {gen_entities}")
        logger.debug(f"Reference entities ({len(ref_entities)}): {ref_entities}")

        if not ref_entities:
            logger.debug("No reference entities found, returning 1.0")
            return 1.0

        # Calculate overlap
        overlap = len(ref_entities.intersection(gen_entities))
        common_entities = ref_entities.intersection(gen_entities)
        logger.debug(f"Common entities ({overlap}): {common_entities}")
        recall = overlap / len(ref_entities)
        logger.debug(f"Article Entity Recall: {overlap}/{len(ref_entities)} = {recall}")
        return recall
        """
        Calculate Article Entity Recall (AER) using simple entity extraction.

        This gives a good approximation of factual content overlap
        without complex NLP dependencies.
        """
        gen_entities = self.extract_entities(generated)
        ref_entities = self.extract_entities(reference)

        if not ref_entities:
            return 1.0

        # Calculate overlap
        overlap = len(ref_entities.intersection(gen_entities))
        return overlap / len(ref_entities)
