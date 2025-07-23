# src/evaluation/metrics/rouge_metrics.py
from collections import Counter

import re
from typing import Dict, List


class ROUGEMetrics:
    """
    ROUGE implementation with smart preprocessing.
    Uses adaptive text processing instead of hardcoded rules.
    """

    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        """
        STORM-aligned text preprocessing with minimal modifications.
        """
        # Convert to lowercase (STORM standard)
        text = text.lower()

        # More conservative punctuation handling (preserve important structure)
        text = re.sub(r"[^\w\s\-\.]", " ", text)

        # Split into words
        words = text.split()

        # Minimal filtering (closer to STORM approach)
        filtered_words = []
        for word in words:
            # Skip very short words
            if len(word) < 2:
                continue

            # Keep years and significant numbers (STORM approach)
            if word.isdigit():
                if len(word) == 4 and word.startswith(("19", "20")):  # Years
                    filtered_words.append(word)
                elif len(word) >= 3:  # Other significant numbers
                    filtered_words.append(word)
                continue

            # Normalize hyphenated words
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

    def calculate_all_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate all ROUGE metrics efficiently."""
        return {
            "rouge_1": self.calculate_rouge_1(generated, reference),
            "rouge_l": self.calculate_rouge_l(generated, reference),
        }
