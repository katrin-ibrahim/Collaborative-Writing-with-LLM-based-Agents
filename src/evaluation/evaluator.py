import logging
from typing import Dict, Any
from utils.data_models import Article, EvaluationResult
from evaluation.benchmarks.freshwiki_loader import FreshWikiEntry

logger = logging.getLogger(__name__)

class ArticleEvaluator:
    """
    Basic evaluator for comparing generated articles against reference content.
    
    This implementation provides essential metrics for baseline comparison
    while maintaining the structure for more sophisticated evaluation later.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def evaluate_article(self, article: Article, reference: FreshWikiEntry) -> Dict[str, float]:
        """
        Evaluate generated article against reference content.
        
        Returns basic but meaningful metrics that allow comparison between
        different generation approaches and track improvement over time.
        """
        try:
            metrics = {}
            
            # Content-based metrics
            metrics['content_length_ratio'] = self._calculate_length_ratio(article.content, reference.reference_content)
            metrics['section_coverage'] = self._calculate_section_coverage(article, reference)
            
            # Basic ROUGE-like overlap (simplified implementation)
            metrics['word_overlap'] = self._calculate_word_overlap(article.content, reference.reference_content)
            
            # Structure similarity
            metrics['heading_similarity'] = self._calculate_heading_similarity(article, reference)
            
            self.logger.info(f"Evaluation completed for article: {article.title}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return {"evaluation_error": 1.0}
    
    def _calculate_length_ratio(self, generated: str, reference: str) -> float:
        """Calculate how the generated content length compares to reference."""
        if not reference:
            return 0.0
        gen_words = len(generated.split())
        ref_words = len(reference.split())
        # Return ratio capped at 2.0 to handle very long generated content
        return min(gen_words / ref_words, 2.0) if ref_words > 0 else 0.0
    
    def _calculate_section_coverage(self, article: Article, reference: FreshWikiEntry) -> float:
        """Calculate what portion of reference topics are covered."""
        if not reference.reference_outline:
            return 0.5  # Default score when no reference outline available
        
        article_sections = set(article.sections.keys()) if article.sections else set()
        reference_sections = set(reference.reference_outline)
        
        if not reference_sections:
            return 0.5
        
        # Calculate overlap between section topics (case-insensitive)
        article_sections_lower = {s.lower() for s in article_sections}
        reference_sections_lower = {s.lower() for s in reference_sections}
        
        overlap = len(article_sections_lower.intersection(reference_sections_lower))
        return overlap / len(reference_sections_lower)
    
    def _calculate_word_overlap(self, generated: str, reference: str) -> float:
        """Calculate basic word-level overlap between generated and reference content."""
        if not generated or not reference:
            return 0.0
        
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        
        if not ref_words:
            return 0.0
        
        overlap = len(gen_words.intersection(ref_words))
        return overlap / len(ref_words)
    
    def _calculate_heading_similarity(self, article: Article, reference: FreshWikiEntry) -> float:
        """Calculate similarity between generated and reference headings."""
        if not reference.reference_outline:
            return 0.5
        
        if article.outline:
            gen_headings = article.outline.headings
        elif article.sections:
            gen_headings = list(article.sections.keys())
        else:
            return 0.0
        
        # Simple heading similarity based on word overlap
        total_similarity = 0.0
        for ref_heading in reference.reference_outline:
            best_match = 0.0
            ref_words = set(ref_heading.lower().split())
            
            for gen_heading in gen_headings:
                gen_words = set(gen_heading.lower().split())
                if ref_words and gen_words:
                    overlap = len(ref_words.intersection(gen_words))
                    similarity = overlap / len(ref_words.union(gen_words))
                    best_match = max(best_match, similarity)
            
            total_similarity += best_match
        
        return total_similarity / len(reference.reference_outline) if reference.reference_outline else 0.0