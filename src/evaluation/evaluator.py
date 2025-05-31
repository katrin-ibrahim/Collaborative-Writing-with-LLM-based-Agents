# src/evaluation/evaluator.py - Clean orchestration with modular metrics
import logging
from typing import Dict, Any
from utils.data_models import Article, EvaluationResult
from evaluation.benchmarks.freshwiki_loader import FreshWikiEntry
from evaluation.metrics.rouge_metrics import ROUGEMetrics
from evaluation.metrics.entity_metrics import EntityMetrics
from evaluation.metrics.heading_metrics import HeadingMetrics

logger = logging.getLogger(__name__)

class ArticleEvaluator:
    """
    Clean evaluator that orchestrates modular metric components.
    
    This design separates concerns:
    - Each metric is independently testable
    - Easy to add new metrics for collaboration research
    - Clear interfaces for each evaluation dimension
    - Simple to debug when metrics fail
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize metric calculators
        self.rouge_metrics = ROUGEMetrics()
        self.entity_metrics = EntityMetrics()
        self.heading_metrics = HeadingMetrics()
        
        self.logger.info("ArticleEvaluator initialized with modular metrics")
    
    def evaluate_article(self, article: Article, reference: FreshWikiEntry) -> Dict[str, float]:
        """
        Comprehensive evaluation using modular metric components.
        
        Each metric component is responsible for its own domain:
        - ROUGE: Content overlap and fluency
        - Entity: Factual accuracy and completeness
        - Heading: Structure and topic coverage
        """
        try:
            metrics = {}
            
            # 1. Content Overlap Metrics (ROUGE family)
            rouge_scores = self.rouge_metrics.calculate_all_rouge(
                article.content, reference.reference_content
            )
            metrics.update(rouge_scores)
            
            # 2. Structural Metrics (Storm HSR, coverage, etc.)
            heading_scores = self.heading_metrics.analyze_heading_quality(
                article.content, reference.reference_outline
            )
            metrics.update(heading_scores)
            
            # 3. Factual Content Metrics (Storm AER, HER)
            entity_scores = self.entity_metrics.calculate_entity_recall(
                article.content, reference.reference_content
            )
            metrics.update(entity_scores)
            
            # 4. Overall Entity Recall (AER from Storm)
            metrics['article_entity_recall'] = self.entity_metrics.calculate_overall_entity_recall(
                article.content, reference.reference_content
            )
            
            # 5. Research-Specific Storm Metrics
            storm_metrics = self._calculate_storm_specific_metrics(article, reference)
            metrics.update(storm_metrics)
            
            # 6. Content Quality Indicators
            quality_metrics = self._calculate_content_quality(article, reference)
            metrics.update(quality_metrics)
            
            self.logger.info(f"Evaluation completed with {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {"evaluation_error": 1.0}
    
    def _calculate_storm_specific_metrics(self, article: Article, reference: FreshWikiEntry) -> Dict[str, float]:
        """
        Calculate the specific metrics from the Storm paper.
        
        These are the exact metrics mentioned in your research paper for RQ1/RQ2.
        """
        metrics = {}
        
        # Extract headings for heading-specific analysis
        generated_headings = self.heading_metrics.extract_headings_from_content(article.content)
        
        # HSR: Heading Soft Recall (already calculated in heading_metrics as heading_soft_recall)
        # This measures topic coverage using semantic similarity
        
        # HER: Heading Entity Recall (entities specifically in headings)
        if reference.reference_outline and generated_headings:
            heading_text_ref = ' '.join(reference.reference_outline)
            heading_text_gen = ' '.join(generated_headings)
            
            # Calculate entity recall specifically for headings
            ref_entities = self.entity_metrics.extract_entities(heading_text_ref)
            gen_entities = self.entity_metrics.extract_entities(heading_text_gen)
            
            all_ref_heading_entities = set()
            all_gen_heading_entities = set()
            
            for entity_type in ref_entities:
                all_ref_heading_entities.update(ref_entities[entity_type])
                all_gen_heading_entities.update(gen_entities[entity_type])
            
            if all_ref_heading_entities:
                metrics['heading_entity_recall'] = (
                    len(all_ref_heading_entities.intersection(all_gen_heading_entities)) 
                    / len(all_ref_heading_entities)
                )
            else:
                metrics['heading_entity_recall'] = 1.0
        else:
            metrics['heading_entity_recall'] = 0.0
        
        return metrics
    
    def _calculate_content_quality(self, article: Article, reference: FreshWikiEntry) -> Dict[str, float]:
        """Calculate general content quality metrics."""
        gen_words = len(article.content.split())
        ref_words = len(reference.reference_content.split()) if reference.reference_content else 0
        
        return {
            'content_length_ratio': min(gen_words / ref_words, 2.0) if ref_words > 0 else 0.0,
            'content_word_count': gen_words
        }
    
    def evaluate_outline_only(self, outline_headings: list, reference: FreshWikiEntry) -> Dict[str, float]:
        """
        Evaluate outline quality separately from full article.
        
        This supports iterative evaluation during the writing process
        and is useful for your collaboration research.
        """
        return self.heading_metrics.analyze_heading_quality(
            '\n'.join([f'## {h}' for h in outline_headings]), 
            reference.reference_outline
        )
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available metrics for research analysis."""
        return {
            'rouge_1': 'Unigram overlap - measures content similarity',
            'rouge_2': 'Bigram overlap - measures fluency and coherence', 
            'rouge_l': 'Longest common subsequence - measures structural similarity',
            'heading_soft_recall': 'Storm HSR - semantic topic coverage',
            'heading_entity_recall': 'Storm HER - named entity coverage in headings',
            'article_entity_recall': 'Storm AER - overall factual content coverage',
            'heading_coverage': 'Percentage of reference topics covered',
            'structure_similarity': 'Structural organization quality',
            'content_length_ratio': 'Generated vs reference content length'
        }