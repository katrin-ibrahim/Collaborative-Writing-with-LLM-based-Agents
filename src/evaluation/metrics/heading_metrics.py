# src/evaluation/metrics/heading_metrics.py
from typing import List, Dict, Set
import re
from collections import Counter

class HeadingMetrics:
    """
    Heading analysis metrics for structure evaluation.
    
    This addresses the 0.0 section coverage and heading similarity scores
    by implementing proper semantic and structural heading comparison.
    """
    
    def __init__(self):
        # Common heading word categories for semantic matching
        self.heading_synonyms = {
            'introduction': ['intro', 'overview', 'background', 'about', 'getting started'],
            'methods': ['methodology', 'approach', 'techniques', 'process', 'implementation'],
            'results': ['findings', 'outcomes', 'analysis', 'evaluation', 'performance'],
            'discussion': ['implications', 'interpretation', 'significance', 'meaning'],
            'conclusion': ['summary', 'closing', 'final thoughts', 'wrap-up', 'takeaways'],
            'history': ['background', 'origins', 'development', 'evolution', 'timeline'],
            'applications': ['uses', 'examples', 'implementation', 'practical', 'real-world'],
            'challenges': ['problems', 'issues', 'limitations', 'difficulties', 'obstacles'],
            'future': ['outlook', 'prospects', 'trends', 'developments', 'next steps']
        }
    
    def normalize_heading(self, heading: str) -> str:
        """
        Normalize heading for better comparison.
        
        Current 0.0 heading similarity suggests normalization issues.
        """
        # Remove common prefixes/suffixes
        heading = re.sub(r'^(?:\d+\.?\s*|\w+\.\s*)', '', heading)  # Remove numbering
        heading = re.sub(r'[^\w\s]', ' ', heading.lower())  # Remove punctuation, lowercase
        heading = ' '.join(heading.split())  # Normalize whitespace
        return heading.strip()
    
    def extract_headings_from_content(self, content: str) -> List[str]:
        """
        Extract headings from markdown-style content.
        
        This fixes the issue where headings aren't being properly extracted
        from generated content.
        """
        headings = []
        
        # Extract markdown headings
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Remove markdown symbols and extract heading text
                heading = re.sub(r'^#+\s*', '', line).strip()
                if heading:
                    headings.append(heading)
        
        return headings
    
    def calculate_semantic_similarity(self, heading1: str, heading2: str) -> float:
        """
        Calculate semantic similarity between two headings.
        
        Uses word overlap and synonym matching for better similarity detection.
        """
        norm1 = self.normalize_heading(heading1)
        norm2 = self.normalize_heading(heading2)
        
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        # Direct word overlap
        direct_overlap = len(words1.intersection(words2))
        total_words = len(words1.union(words2))
        
        if total_words == 0:
            return 0.0
        
        direct_score = direct_overlap / total_words
        
        # Synonym-based overlap
        synonym_score = 0.0
        synonym_matches = 0
        
        for word1 in words1:
            for word2 in words2:
                for category, synonyms in self.heading_synonyms.items():
                    if word1 in synonyms and word2 in synonyms:
                        synonym_matches += 1
                        break
                    elif word1 == category and word2 in synonyms:
                        synonym_matches += 1
                        break
                    elif word2 == category and word1 in synonyms:
                        synonym_matches += 1
                        break
        
        if total_words > 0:
            synonym_score = synonym_matches / total_words
        
        # Combine direct and synonym scores
        return max(direct_score, synonym_score * 0.8)  # Weight synonyms slightly lower
    
    def calculate_heading_soft_recall(self, generated_headings: List[str], 
                                    reference_headings: List[str]) -> float:
        """
        Calculate Heading Soft Recall (HSR) using semantic similarity.
        
        This implements the research metric for topic coverage evaluation.
        """
        if not reference_headings:
            return 1.0
        
        if not generated_headings:
            return 0.0
        
        total_similarity = 0.0
        
        for ref_heading in reference_headings:
            best_similarity = 0.0
            
            for gen_heading in generated_headings:
                similarity = self.calculate_semantic_similarity(ref_heading, gen_heading)
                best_similarity = max(best_similarity, similarity)
            
            total_similarity += best_similarity
        
        return total_similarity / len(reference_headings)
    
    def calculate_heading_coverage(self, generated_headings: List[str], 
                                 reference_headings: List[str]) -> float:
        """
        Calculate what percentage of reference topics are covered.
        
        This fixes the 0.0 section coverage issue by using semantic matching.
        """
        if not reference_headings:
            return 1.0
        
        if not generated_headings:
            return 0.0
        
        covered_topics = 0
        threshold = 0.3  # Similarity threshold for considering a topic "covered"
        
        for ref_heading in reference_headings:
            for gen_heading in generated_headings:
                similarity = self.calculate_semantic_similarity(ref_heading, gen_heading)
                if similarity >= threshold:
                    covered_topics += 1
                    break
        
        return covered_topics / len(reference_headings)
    
    def calculate_structure_similarity(self, generated_headings: List[str], 
                                     reference_headings: List[str]) -> float:
        """
        Calculate structural similarity between heading sequences.
        
        This measures how well the generated structure matches the reference.
        """
        if not generated_headings or not reference_headings:
            return 0.0
        
        # Create position-aware similarity matrix
        similarities = []
        
        for i, ref_heading in enumerate(reference_headings):
            for j, gen_heading in enumerate(generated_headings):
                semantic_sim = self.calculate_semantic_similarity(ref_heading, gen_heading)
                
                # Position penalty: prefer headings in similar positions
                position_diff = abs(i / len(reference_headings) - j / len(generated_headings))
                position_penalty = 1.0 - position_diff
                
                combined_score = semantic_sim * 0.7 + position_penalty * 0.3
                similarities.append(combined_score)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def analyze_heading_quality(self, generated_content: str, 
                               reference_headings: List[str]) -> Dict[str, float]:
        """
        Comprehensive heading analysis.
        
        This provides all heading-related metrics needed for evaluation.
        """
        generated_headings = self.extract_headings_from_content(generated_content)
        
        return {
            'heading_soft_recall': self.calculate_heading_soft_recall(
                generated_headings, reference_headings
            ),
            'heading_coverage': self.calculate_heading_coverage(
                generated_headings, reference_headings
            ),
            'structure_similarity': self.calculate_structure_similarity(
                generated_headings, reference_headings
            ),
            'heading_count_ratio': len(generated_headings) / len(reference_headings) 
                                 if reference_headings else 0.0
        }