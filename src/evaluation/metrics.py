# src/evaluation/metrics.py
"""
Consolidated metrics functions for article evaluation.

This module provides all STORM paper metric calculations as standalone functions.

All metrics are implemented as pure functions for easy testing and reuse.
"""

from collections import Counter

import logging
import numpy as np
import re

# External dependencies for NLP functionality
from flair.data import Sentence
from flair.models import SequenceTagger
from sentence_transformers import SentenceTransformer
from typing import Dict, List

logger = logging.getLogger(__name__)

_NER_TAGGER = None
_SENTENCE_MODEL = None

# ============================================================================
# Constants
# ============================================================================

STORM_METRICS = [
    "rouge_1",
    "rouge_l",
    "heading_soft_recall",
    "heading_entity_recall",
    "article_entity_recall",
]

METRIC_DESCRIPTIONS = {
    "rouge_1": "STORM: Unigram overlap between generated and reference content (0-100%)",
    "rouge_l": "STORM: Longest common subsequence overlap (0-100%)",
    "heading_soft_recall": "STORM HSR: Semantic topic coverage in headings (0-100%)",
    "heading_entity_recall": "STORM HER: Entity coverage in headings only (0-100%)",
    "article_entity_recall": "STORM AER: Overall factual content coverage (0-100%)",
}


# ============================================================================
# Model Loading Functions
# ============================================================================


def _get_ner_tagger():
    """Get or load the NER tagger with proper caching."""
    global _NER_TAGGER
    if _NER_TAGGER is not None:
        return _NER_TAGGER

    try:
        _NER_TAGGER = SequenceTagger.load("ner")
        logger.info("FLAIR NER model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load FLAIR NER model: {e}")
        _NER_TAGGER = None

    return _NER_TAGGER


def _get_sentence_model():
    """Get or load the sentence transformer model with proper caching."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is not None:
        return _SENTENCE_MODEL

    try:
        _SENTENCE_MODEL = SentenceTransformer(
            "all-MiniLM-L6-v2", device="cpu"
        )  # Use "cuda" if available
        logger.info("Sentence transformer model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load sentence transformer: {e}")
        _SENTENCE_MODEL = None

    return _SENTENCE_MODEL


# ============================================================================
# Text Preprocessing Functions
# ============================================================================


def preprocess_text_for_rouge(text: str) -> List[str]:
    """STORM-aligned text preprocessing for ROUGE metrics."""
    text = text.lower()
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    words = text.split()

    filtered_words = []
    for word in words:
        if len(word) < 2:
            continue

        if word.isdigit():
            if len(word) == 4 and word.startswith(("19", "20")):
                filtered_words.append(word)
            elif len(word) >= 3:
                filtered_words.append(word)
            continue

        if "-" in word:
            word = word.replace("-", "_")

        filtered_words.append(word)

    return filtered_words


# ============================================================================
# ROUGE Metrics Functions
# ============================================================================


def calculate_rouge_1(generated: str, reference: str) -> float:
    """Calculate ROUGE-1 (unigram overlap) with STORM preprocessing."""
    gen_words = preprocess_text_for_rouge(generated)
    ref_words = preprocess_text_for_rouge(reference)

    if not ref_words:
        return 0.0

    gen_counter = Counter(gen_words)
    ref_counter = Counter(ref_words)

    overlap = sum((gen_counter & ref_counter).values())
    return overlap / len(ref_words)


def calculate_rouge_l(generated: str, reference: str) -> float:
    """Calculate ROUGE-L (longest common subsequence) efficiently."""
    gen_words = preprocess_text_for_rouge(generated)
    ref_words = preprocess_text_for_rouge(reference)

    if not gen_words or not ref_words:
        return 0.0

    # Efficient LCS using dynamic programming
    m, n = len(gen_words), len(ref_words)
    prev_row = [0] * (n + 1)
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gen_words[i - 1] == ref_words[j - 1]:
                curr_row[j] = prev_row[j - 1] + 1
            else:
                curr_row[j] = max(prev_row[j], curr_row[j - 1])
        prev_row, curr_row = curr_row, prev_row

    lcs_length = prev_row[n]
    return lcs_length / len(ref_words)


def calculate_all_rouge_metrics(generated: str, reference: str) -> Dict[str, float]:
    """Calculate all ROUGE metrics efficiently."""
    return {
        "rouge_1": calculate_rouge_1(generated, reference),
        "rouge_l": calculate_rouge_l(generated, reference),
    }


# ============================================================================
# Entity Extraction Functions
# ============================================================================


def extract_entities(text: str) -> set:
    """Extract named entities using FLAIR with STORM confidence threshold."""
    ner_tagger = _get_ner_tagger()

    if ner_tagger is None:
        logger.warning("NER tagger not available, returning empty entity set")
        return set()

    entities = set()

    try:
        sentence = Sentence(text)
        ner_tagger.predict(sentence)

        for entity in sentence.get_spans("ner"):
            if entity.score > 0.5:  # STORM confidence threshold
                entity_text = entity.text.lower().strip()
                if len(entity_text) > 1:
                    entities.add(entity_text)
                    logger.debug(
                        f"FLAIR entity: {entity_text} ({entity.tag}, {entity.score:.3f})"
                    )

    except Exception as e:
        logger.warning(f"FLAIR NER failed: {e}. returning empty set.")

    return entities


def calculate_entity_recall(generated: str, reference: str) -> float:
    """Calculate Article Entity Recall (AER) using STORM-compliant entity extraction."""
    logger.debug("=== Article Entity Recall Calculation ===")

    try:
        gen_entities = extract_entities(generated)
        ref_entities = extract_entities(reference)

        logger.debug(f"Generated entities ({len(gen_entities)}): {gen_entities}")
        logger.debug(f"Reference entities ({len(ref_entities)}): {ref_entities}")

        if not ref_entities:
            logger.debug("No reference entities found, returning 1.0")
            return 1.0

        overlap = len(ref_entities.intersection(gen_entities))
        common_entities = ref_entities.intersection(gen_entities)
        logger.debug(f"Common entities ({overlap}): {common_entities}")

        recall = overlap / len(ref_entities)
        logger.debug(f"Article Entity Recall: {overlap}/{len(ref_entities)} = {recall}")
        return recall

    except Exception as e:
        logger.error(f"Entity recall calculation failed: {e}")
        return 0.0


# ============================================================================
# Heading Extraction and Similarity Functions
# ============================================================================


def extract_headings_from_content(content: str) -> List[str]:
    """Extract headings from markdown-style content."""
    headings = []

    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        # Match markdown headers (# ## ###) and common heading patterns
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if heading:
                headings.append(heading)
        elif line.endswith(":") and len(line) < 100:
            # Potential section heading
            headings.append(line.rstrip(":"))

    logger.debug(f"Extracted headings: {headings}")
    return headings


def calculate_heading_soft_recall(
    generated_headings: List[str], reference_headings: List[str]
) -> float:
    """Calculate Heading Soft Recall (HSR) using semantic similarity."""
    if not reference_headings or not generated_headings:
        logger.debug("Missing headings, returning 0.0")
        return 0.0

    sentence_model = _get_sentence_model()
    if sentence_model is None:
        logger.warning("Sentence transformer not available, returning 0.0")
        return 0.0

    try:
        # Generate embeddings
        ref_embeddings = sentence_model.encode(reference_headings)
        gen_embeddings = sentence_model.encode(generated_headings)

        # Calculate cosine similarities
        similarities = []
        for ref_emb in ref_embeddings:
            max_sim = 0.0
            for gen_emb in gen_embeddings:
                similarity = np.dot(ref_emb, gen_emb) / (
                    np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb)
                )
                max_sim = max(max_sim, similarity)
            similarities.append(max_sim)

        hsr_score = np.mean(similarities)
        logger.debug(f"Heading Soft Recall: {hsr_score}")
        return hsr_score

    except Exception as e:
        logger.error(f"HSR calculation failed: {e}")
        return 0.0


def calculate_heading_entity_recall(
    generated_headings: List[str], reference_headings: List[str]
) -> float:
    """Calculate Heading Entity Recall (HER) using entity extraction on headings only."""
    logger.debug("=== Heading Entity Recall Calculation ===")
    logger.debug(f"Generated headings: {generated_headings}")
    logger.debug(f"Reference headings: {reference_headings}")

    if not reference_headings or not generated_headings:
        logger.debug("Missing headings, returning 0.0")
        return 0.0

    # Extract entities from each heading individually
    ref_entities = set()
    for heading in reference_headings:
        heading_entities = extract_entities(heading)
        ref_entities.update(heading_entities)
        logger.debug(f"Reference heading '{heading}' entities: {heading_entities}")

    gen_entities = set()
    for heading in generated_headings:
        heading_entities = extract_entities(heading)
        gen_entities.update(heading_entities)
        logger.debug(f"Generated heading '{heading}' entities: {heading_entities}")

    logger.debug(f"All reference heading entities: {ref_entities}")
    logger.debug(f"All generated heading entities: {gen_entities}")

    if not ref_entities:
        logger.debug("No reference heading entities, returning 0.0 (undefined HER)")
        return 0.0

    # Calculate overlap
    overlap = len(ref_entities.intersection(gen_entities))
    common_entities = ref_entities.intersection(gen_entities)
    logger.debug(f"Common heading entities ({overlap}): {common_entities}")
    recall = overlap / len(ref_entities)
    logger.debug(f"Heading Entity Recall: {overlap}/{len(ref_entities)} = {recall}")
    return recall


# ============================================================================
# Combined Metrics Functions
# ============================================================================


def calculate_heading_metrics(
    generated_content: str, reference_headings: List[str]
) -> Dict[str, float]:
    """Calculate both heading metrics (HSR and HER) from content and reference headings."""
    generated_headings = extract_headings_from_content(generated_content)

    hsr = calculate_heading_soft_recall(generated_headings, reference_headings)
    her = calculate_heading_entity_recall(generated_headings, reference_headings)

    return {
        "heading_soft_recall": hsr,
        "heading_entity_recall": her,
        "extracted_headings": generated_headings,
    }


def evaluate_article_metrics(
    article_content: str, reference_content: str, reference_headings: List[str]
) -> Dict[str, float]:
    """
    Calculate all STORM metrics for an article.

    Args:
        article_content: Generated article text
        reference_content: Reference article text
        reference_headings: List of reference headings

    Returns:
        Dictionary with all STORM metrics (0-100 scale)
    """
    try:
        metrics = {}

        # 1. ROUGE Metrics (Content Overlap)
        rouge_scores = calculate_all_rouge_metrics(article_content, reference_content)
        for metric, score in rouge_scores.items():
            metrics[metric] = score * 100.0

        # 2. Heading Metrics
        heading_results = calculate_heading_metrics(article_content, reference_headings)
        metrics["heading_soft_recall"] = heading_results["heading_soft_recall"] * 100.0
        metrics["heading_entity_recall"] = (
            heading_results["heading_entity_recall"] * 100.0
        )

        # 3. Article Entity Recall (AER)
        aer_score = calculate_entity_recall(article_content, reference_content)
        metrics["article_entity_recall"] = aer_score * 100.0

        logger.debug(f"Article evaluation completed: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Article evaluation failed: {e}")
        return {
            "rouge_1": 0.0,
            "rouge_l": 0.0,
            "heading_soft_recall": 0.0,
            "heading_entity_recall": 0.0,
            "article_entity_recall": 0.0,
        }


# ============================================================================
# Utility Functions
# ============================================================================


def format_metrics_for_display(
    metrics: Dict[str, float], precision: int = 2
) -> Dict[str, str]:
    """Format metrics for human-readable display."""
    formatted = {}
    for metric_name, value in metrics.items():
        if metric_name in STORM_METRICS:
            formatted[metric_name] = f"{value:.{precision}f}%"
        else:
            formatted[metric_name] = f"{value:.{precision}f}"
    return formatted


def calculate_composite_score(
    metrics: Dict[str, float], weights: Dict[str, float] = None
) -> float:
    """Calculate a weighted composite score from multiple metrics."""
    if weights is None:
        weights = {metric: 1.0 for metric in STORM_METRICS}

    weighted_sum = 0.0
    total_weight = 0.0

    for metric_name, weight in weights.items():
        if metric_name in metrics:
            weighted_sum += metrics[metric_name] * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0
