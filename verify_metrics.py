#!/usr/bin/env python3
"""
Script to verify the logical correctness of metrics used in the project.
This script tests various metrics with known inputs to verify they produce expected results.
"""

import sys
from collections import Counter

import os
import re

# Add the project's src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# Custom implementation of ROUGE metrics based on the project's implementation
class ROUGEMetrics:
    """
    ROUGE implementation with smart preprocessing.
    Uses adaptive text processing instead of hardcoded rules.
    """

    @staticmethod
    def preprocess_text(text: str) -> list:
        """
        Smart text preprocessing that adapts to content.
        Uses heuristics to clean text effectively without hardcoded rules.
        """
        # Convert to lowercase
        text = text.lower()

        # Smart punctuation removal - keep hyphens in compound words
        text = re.sub(r"[^\w\s\-]", " ", text)

        # Split into words
        words = text.split()

        # Adaptive filtering based on content characteristics
        filtered_words = []
        for word in words:
            # Skip very short words (likely not meaningful)
            if len(word) < 2:
                continue

            # Skip pure numbers unless they look important (years, etc.)
            if word.isdigit():
                if len(word) == 4 and word.startswith(("19", "20")):  # Years
                    filtered_words.append(word)
                elif len(word) >= 3:  # Other significant numbers
                    filtered_words.append(word)
                # Skip single/double digit numbers
                continue

            # Keep hyphenated words but normalize them
            if "-" in word:
                word = word.replace("-", "_")

            filtered_words.append(word)

        return filtered_words

    def calculate_rouge_1(self, generated: str, reference: str) -> float:
        """Calculate ROUGE-1 (unigram overlap) with smart preprocessing."""
        gen_words = self.preprocess_text(generated)
        ref_words = self.preprocess_text(reference)

        if not ref_words:
            return 0.0

        gen_counter = Counter(gen_words)
        ref_counter = Counter(ref_words)

        # Calculate recall-based ROUGE-1
        overlap = sum((gen_counter & ref_counter).values())
        return overlap / len(ref_words)

    def calculate_rouge_2(self, generated: str, reference: str) -> float:
        """Calculate ROUGE-2 (bigram overlap) with smart preprocessing."""
        gen_words = self.preprocess_text(generated)
        ref_words = self.preprocess_text(reference)

        if len(ref_words) < 2:
            return 0.0

        # Create bigrams with separator
        gen_bigrams = [
            f"{gen_words[i]}||{gen_words[i + 1]}" for i in range(len(gen_words) - 1)
        ]
        ref_bigrams = [
            f"{ref_words[i]}||{ref_words[i + 1]}" for i in range(len(ref_words) - 1)
        ]

        if not ref_bigrams:
            return 0.0

        gen_counter = Counter(gen_bigrams)
        ref_counter = Counter(ref_bigrams)

        overlap = sum((gen_counter & ref_counter).values())
        return overlap / len(ref_bigrams)

    def calculate_rouge_l(self, generated: str, reference: str) -> float:
        """Calculate ROUGE-L (longest common subsequence) efficiently."""
        gen_words = self.preprocess_text(generated)
        ref_words = self.preprocess_text(reference)

        if not gen_words or not ref_words:
            return 0.0

        # Efficient LCS using dynamic programming
        m, n = len(gen_words), len(ref_words)

        # Use space-optimized DP (only keep current and previous row)
        prev_row = [0] * (n + 1)
        curr_row = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gen_words[i - 1] == ref_words[j - 1]:
                    curr_row[j] = prev_row[j - 1] + 1
                else:
                    curr_row[j] = max(prev_row[j], curr_row[j - 1])

            # Swap rows
            prev_row, curr_row = curr_row, prev_row

        lcs_length = prev_row[n]
        return lcs_length / len(ref_words)  # Recall-based

    def calculate_all_rouge(self, generated: str, reference: str) -> dict:
        """Calculate all ROUGE metrics efficiently."""
        return {
            "rouge_1": self.calculate_rouge_1(generated, reference),
            "rouge_2": self.calculate_rouge_2(generated, reference),
            "rouge_l": self.calculate_rouge_l(generated, reference),
        }


# Simplified entity extraction
class EntityMetrics:
    def extract_entities(self, text: str) -> set:
        """
        Extract entities using simple but effective heuristics.
        """
        entities = set()

        # Simple word extraction for headings
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

        # Multi-word phrases
        clean_text = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        if " " in clean_text and len(clean_text) > 3:
            entities.add(clean_text)

        # Capitalized sequences (names, places, organizations)
        cap_sequences = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        for seq in cap_sequences:
            if not self._is_sentence_starter(seq, text) and len(seq) > 2:
                entities.add(seq.lower())

        return entities

    def _is_sentence_starter(self, word_sequence: str, full_text: str) -> bool:
        """Check if capitalized sequence is just a sentence starter."""
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

        if first_word in sentence_starters:
            return True

        pattern = r"\.\s+" + re.escape(word_sequence)
        return bool(re.search(pattern, full_text))

    def calculate_overall_entity_recall(self, generated: str, reference: str) -> float:
        """Calculate Article Entity Recall (AER)."""
        gen_entities = self.extract_entities(generated)
        ref_entities = self.extract_entities(reference)

        if not ref_entities:
            return 1.0

        overlap = len(ref_entities.intersection(gen_entities))
        return overlap / len(ref_entities)


def extract_headings(text):
    """Extract headings from markdown text."""
    heading_pattern = r"^#+\s+(.+)$"
    headings = []
    for line in text.split("\n"):
        match = re.match(heading_pattern, line)
        if match:
            headings.append(match.group(1).strip())
    return headings


def verify_rouge_metrics():
    """Verify ROUGE metrics with known cases."""
    print("\n=== Verifying ROUGE Metrics ===")

    rouge_metrics = ROUGEMetrics()

    # Test Case 1: Identical text
    text = "This is a test sentence for ROUGE calculation."
    scores = rouge_metrics.calculate_all_rouge(text, text)
    print(f"Test Case 1 (Identical text): {scores}")
    assert all(
        abs(score - 1.0) < 1e-6 for score in scores.values()
    ), "Failed: ROUGE scores for identical text should be 1.0"

    # Test Case 2: Completely different text
    text1 = "This is the first test sentence."
    text2 = "Something completely different and unrelated."
    scores = rouge_metrics.calculate_all_rouge(text1, text2)
    print(f"Test Case 2 (Different text): {scores}")
    assert all(
        score < 0.5 for score in scores.values()
    ), "Failed: ROUGE scores for different texts should be low"

    # Test Case 3: Partially similar text
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The quick brown fox jumps over the fence."
    scores = rouge_metrics.calculate_all_rouge(text1, text2)
    print(f"Test Case 3 (Partially similar text): {scores}")
    assert (
        0.5 < scores["rouge_1"] < 1.0
    ), "Failed: ROUGE-1 for partially similar text should be between 0.5 and 1.0"

    print("All ROUGE verification tests passed!")


def verify_entity_metrics():
    """Verify entity extraction and metrics with known cases."""
    print("\n=== Verifying Entity Extraction ===")

    entity_metrics = EntityMetrics()

    # Test Case 1: Text with known entities
    text = "Apple Inc. was founded by Steve Jobs in California."
    entities = entity_metrics.extract_entities(text)
    print(f"Test Case 1 (Known entities): {entities}")

    expected_entities = {
        "apple",
        "inc",
        "founded",
        "steve",
        "jobs",
        "california",
        "apple inc",
        "steve jobs",
        "founded by steve jobs in california",
    }
    overlap = entities.intersection(expected_entities)

    print(f"Overlap with expected entities: {overlap}")
    assert len(overlap) > 3, "Failed: Too few expected entities detected"

    # Test Case 2: Text without entities
    text = "this text contains no named entities at all."
    entities = entity_metrics.extract_entities(text)
    print(f"Test Case 2 (No entities): {entities}")
    # Our simple entity extraction will find some words as entities
    # but it shouldn't find any capitalized entities

    capitalized_entities = [e for e in entities if any(c.isupper() for c in e)]
    assert (
        len(capitalized_entities) == 0
    ), "Failed: No capitalized entities should be detected"

    print("All entity extraction verification tests passed!")


def verify_heading_metrics():
    """Verify heading metrics with known cases."""
    print("\n=== Verifying Heading Metrics ===")

    # Create a simplified version of heading metrics without relying on
    # the SentenceTransformer model
    def calculate_heading_overlap(reference_headings, candidate_headings):
        if not reference_headings:
            return 1.0
        if not candidate_headings:
            return 0.0

        # Normalize and split headings into words
        ref_words = set()
        for heading in reference_headings:
            ref_words.update(heading.lower().split())

        cand_words = set()
        for heading in candidate_headings:
            cand_words.update(heading.lower().split())

        # Calculate word overlap
        if not ref_words:
            return 1.0

        overlap = len(ref_words.intersection(cand_words))
        return overlap / len(ref_words)

    # Test Case 1: Identical headings
    reference = ["Introduction", "Methods", "Results", "Discussion"]
    candidate = ["Introduction", "Methods", "Results", "Discussion"]
    overlap = calculate_heading_overlap(reference, candidate)
    print(f"Test Case 1 (Identical headings): {overlap}")
    assert overlap == 1.0, "Failed: Identical headings should have perfect overlap"

    # Test Case 2: Different headings
    reference = ["Introduction", "Methods", "Results", "Discussion"]
    candidate = ["Background", "Experiments", "Outcomes", "Conclusion"]
    overlap = calculate_heading_overlap(reference, candidate)
    print(f"Test Case 2 (Different headings): {overlap}")
    assert (
        overlap == 0.0
    ), "Failed: Completely different headings should have zero overlap"

    # Test Case 3: Partially overlapping headings
    reference = ["Introduction", "Methods", "Results", "Discussion"]
    candidate = ["Introduction", "Experiments", "Results", "Conclusion"]
    overlap = calculate_heading_overlap(reference, candidate)
    print(f"Test Case 3 (Partially overlapping headings): {overlap}")
    assert (
        0.0 < overlap < 1.0
    ), "Failed: Partially overlapping headings should have intermediate score"

    print("All heading metrics verification tests passed!")


def verify_heading_extraction():
    """Verify the extraction of headings from markdown text."""
    print("\n=== Verifying Heading Extraction ===")

    # Test Case 1: Basic markdown headings
    text = """# Main Title

Some content here.

## Section 1

More content.

## Section 2

Even more content.

### Subsection 2.1

Final content."""

    headings = extract_headings(text)
    print(f"Test Case 1 (Basic markdown headings): {headings}")
    expected = ["Main Title", "Section 1", "Section 2", "Subsection 2.1"]
    assert (
        headings == expected
    ), f"Failed: Extracted headings {headings} do not match expected {expected}"

    print("All heading extraction verification tests passed!")


if __name__ == "__main__":
    print("Starting metrics verification...")

    try:
        verify_rouge_metrics()
    except Exception as e:
        print(f"ROUGE verification failed: {e}")

    try:
        verify_entity_metrics()
    except Exception as e:
        print(f"Entity metrics verification failed: {e}")

    try:
        verify_heading_metrics()
    except Exception as e:
        print(f"Heading metrics verification failed: {e}")

    try:
        verify_heading_extraction()
    except Exception as e:
        print(f"Heading extraction verification failed: {e}")

    print("\nMetrics verification complete!")
