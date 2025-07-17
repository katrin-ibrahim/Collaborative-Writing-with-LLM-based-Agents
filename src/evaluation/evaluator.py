# src/evaluation/simple_evaluator.py
import logging
from typing import Dict

from evaluation.metrics.entity_metrics import EntityMetrics
from evaluation.metrics.heading_metrics import HeadingMetrics
from evaluation.metrics.rouge_metrics import ROUGEMetrics
from utils.data_models import Article
from utils.freshwiki_loader import FreshWikiEntry

logger = logging.getLogger(__name__)


class ArticleEvaluator:
    """
    Evaluator using only STORM paper metrics.

    Uses hybrid approach: smart heuristics + NLP where beneficial.
    Returns exactly the 6 metrics used in STORM paper for clean comparison.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize metric calculators (now with hybrid approach)
        self.rouge_metrics = ROUGEMetrics()
        self.entity_metrics = EntityMetrics()
        self.heading_metrics = HeadingMetrics()

        self.logger.info("SimpleEvaluator initialized with hybrid metrics")

    def evaluate_article(
        self, article: Article, reference: FreshWikiEntry
    ) -> Dict[str, float]:
        """
        Evaluate article using STORM paper metrics only.

        Returns the 6 core STORM metrics in percentage scale (0-100):
        - rouge_1, rouge_2, rouge_l (content overlap)
        - heading_soft_recall (topic coverage)
        - heading_entity_recall (entities in headings)
        - article_entity_recall (overall factual coverage)
        """
        try:
            metrics = {}

            # 1. ROUGE Metrics (Content Overlap) - using smart preprocessing
            rouge_scores = self.rouge_metrics.calculate_all_rouge(
                article.content, reference.reference_content
            )
            # Convert ROUGE scores to percentage scale (0-100)
            for metric, score in rouge_scores.items():
                metrics[metric] = score * 100.0

            # 2. Heading Soft Recall (HSR) - using semantic similarity
            generated_headings = self.heading_metrics.extract_headings_from_content(
                article.content
            )
            hsr_score = self.heading_metrics.calculate_heading_soft_recall(
                generated_headings, reference.reference_outline
            )
            metrics["heading_soft_recall"] = hsr_score * 100.0

            # 3. Heading Entity Recall (HER) - entities specifically in headings
            her_score = self._calculate_heading_entity_recall(
                generated_headings, reference.reference_outline
            )
            metrics["heading_entity_recall"] = her_score * 100.0

            # 4. Article Entity Recall (AER) - overall factual coverage using smart heuristics
            aer_score = self.entity_metrics.calculate_overall_entity_recall(
                article.content, reference.reference_content
            )
            metrics["article_entity_recall"] = aer_score * 100.0

            self.logger.debug(f"Evaluation completed: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            # Return zeros for all STORM metrics on failure (in percentage scale)
            return {
                "rouge_1": 0.0,
                "rouge_2": 0.0,
                "rouge_l": 0.0,
                "heading_soft_recall": 0.0,
                "heading_entity_recall": 0.0,
                "article_entity_recall": 0.0,
            }

    def _calculate_heading_entity_recall(
        self, generated_headings: list, reference_headings: list
    ) -> float:
        """
        Calculate Heading Entity Recall (HER) using entity extraction on headings only.

        This measures how many entities from reference headings appear in generated headings.
        """
        self.logger.debug("=== Heading Entity Recall Calculation ===")
        self.logger.debug(f"Generated headings: {generated_headings}")
        self.logger.debug(f"Reference headings: {reference_headings}")

        if not reference_headings or not generated_headings:
            self.logger.debug("Missing headings, returning 0.0")
            return 0.0

        # Convert heading lists to text for entity extraction
        ref_heading_text = " ".join(reference_headings)
        gen_heading_text = " ".join(generated_headings)

        self.logger.debug(f"Reference heading text: {ref_heading_text}")
        self.logger.debug(f"Generated heading text: {gen_heading_text}")

        # Extract entities from headings only

        # Extract entities from each heading individually to avoid cross-heading extraction
        ref_entities = set()
        for heading in reference_headings:
            heading_entities = self.entity_metrics.extract_entities(heading)
            ref_entities.update(heading_entities)
            self.logger.debug(
                f"Reference heading '{heading}' entities: {heading_entities}"
            )

        gen_entities = set()
        for heading in generated_headings:
            heading_entities = self.entity_metrics.extract_entities(heading)
            gen_entities.update(heading_entities)
            self.logger.debug(
                f"Generated heading '{heading}' entities: {heading_entities}"
            )

        self.logger.debug(f"All reference heading entities: {ref_entities}")
        self.logger.debug(f"All generated heading entities: {gen_entities}")

        if not ref_entities:
            self.logger.debug("No reference heading entities, returning 1.0")
            return 1.0

        # Calculate overlap
        overlap = len(ref_entities.intersection(gen_entities))
        common_entities = ref_entities.intersection(gen_entities)
        self.logger.debug(f"Common heading entities ({overlap}): {common_entities}")
        recall = overlap / len(ref_entities)
        self.logger.debug(
            f"Heading Entity Recall: {overlap}/{len(ref_entities)} = {recall}"
        )
        return recall

    @staticmethod
    def get_metric_descriptions() -> Dict[str, str]:
        """Get descriptions of STORM metrics for documentation."""
        return {
            "rouge_1": "STORM: Unigram overlap between generated and reference content (0-100%)",
            "rouge_2": "STORM: Bigram overlap between generated and reference content (0-100%)",
            "rouge_l": "STORM: Longest common subsequence overlap (0-100%)",
            "heading_soft_recall": "STORM HSR: Semantic topic coverage in headings (0-100%)",
            "heading_entity_recall": "STORM HER: Entity coverage in headings only (0-100%)",
            "article_entity_recall": "STORM AER: Overall factual content coverage (0-100%)",
        }
