# src/evaluation/metrics/entity_metrics.py
from typing import Dict, List, Set
import re

class EntityMetrics:
    """
    Entity-based evaluation metrics using pattern matching.
    
    Since FLAIR may not be available, this uses robust pattern matching
    to identify key entities and measure factual content overlap.
    """
    
    def __init__(self):
        # Patterns for different entity types
        self.entity_patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
                r'\b(?:Mr|Ms|Dr|Prof)\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # Dr. John Smith
            ],
            'ORGANIZATION': [
                r'\b[A-Z][a-zA-Z&\s]+ (?:Inc|Corp|LLC|Ltd|Company|University|Institute)\b',
                r'\b(?:Google|Microsoft|Apple|Amazon|Meta|OpenAI|DeepMind)\b',
            ],
            'LOCATION': [
                r'\b[A-Z][a-z]+ (?:City|State|Country|Province|County)\b',
                r'\b(?:United States|China|Japan|Germany|France|Canada|Australia)\b',
                r'\b[A-Z][a-z]+, [A-Z][a-z]+\b',  # Paris, France
            ],
            'DATE': [
                r'\b\d{4}\b',  # Years
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            ],
            'TECHNOLOGY': [
                r'\b(?:AI|ML|artificial intelligence|machine learning|deep learning|neural network|transformer|GPT|BERT|API|algorithm)\b',
                r'\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b',  # Acronyms like REST-API
            ],
            'NUMBERS': [
                r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)\b',
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Money
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """
        Extract entities using pattern matching.
        
        This provides more reliable entity detection than depending on
        external NER libraries that may not be available.
        """
        entities = {entity_type: set() for entity_type in self.entity_patterns}
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                # Normalize matches
                normalized_matches = {match.lower().strip() for match in matches if match.strip()}
                entities[entity_type].update(normalized_matches)
        
        return entities
    
    def calculate_entity_recall(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate entity recall by type.
        
        This addresses the poor entity coverage suggested by low content scores.
        """
        gen_entities = self.extract_entities(generated)
        ref_entities = self.extract_entities(reference)
        
        recall_scores = {}
        
        for entity_type in self.entity_patterns:
            ref_set = ref_entities[entity_type]
            gen_set = gen_entities[entity_type]
            
            if not ref_set:
                recall_scores[f'{entity_type.lower()}_recall'] = 1.0  # No entities to find
            else:
                overlap = len(ref_set.intersection(gen_set))
                recall_scores[f'{entity_type.lower()}_recall'] = overlap / len(ref_set)
        
        return recall_scores
    
    def calculate_overall_entity_recall(self, generated: str, reference: str) -> float:
        """
        Calculate overall entity recall across all types.
        
        This provides the Article Entity Recall (AER) metric mentioned in the research.
        """
        gen_entities = self.extract_entities(generated)
        ref_entities = self.extract_entities(reference)
        
        # Combine all entity types
        all_ref_entities = set()
        all_gen_entities = set()
        
        for entity_type in self.entity_patterns:
            all_ref_entities.update(ref_entities[entity_type])
            all_gen_entities.update(gen_entities[entity_type])
        
        if not all_ref_entities:
            return 1.0
        
        overlap = len(all_ref_entities.intersection(all_gen_entities))
        return overlap / len(all_ref_entities)
    
    def get_entity_statistics(self, text: str) -> Dict[str, int]:
        """Get statistics about entity distribution in text."""
        entities = self.extract_entities(text)
        return {f'{entity_type.lower()}_count': len(entity_set) 
                for entity_type, entity_set in entities.items()}