# src/evaluation/benchmarks/freshwiki_loader.py - Simplified, evaluation-focused loader
import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class FreshWikiEntry:
    """
    Represents a FreshWiki evaluation benchmark entry.
    
    FreshWiki serves as ground truth for evaluation - it provides reference
    articles that our generated content will be compared against to measure
    quality across multiple dimensions.
    """
    topic: str                              # The topic/title we're generating content for
    reference_outline: List[str]            # Ground truth section headings for structure evaluation
    reference_content: str                  # Full reference article for content comparison
    ground_truth_sections: Dict[str, str]   # Section-by-section ground truth for detailed analysis
    metadata: Dict[str, Any]               # Additional metadata (dates, categories, etc.)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FreshWikiEntry':
        """
        Create FreshWikiEntry from dictionary data.
        
        This method handles the various formats that FreshWiki data might come in,
        extracting the essential components needed for evaluation while being
        tolerant of different field naming conventions.
        """
        return cls(
            topic=data.get('topic', '') or data.get('title', ''),
            reference_outline=data.get('outline', []) or data.get('headings', []),
            reference_content=data.get('content', '') or data.get('text', ''),
            ground_truth_sections=data.get('sections', {}),
            metadata=data.get('metadata', {})
        )

class FreshWikiLoader:
    """
    Loader for FreshWiki evaluation dataset.
    
    This class has a single, focused responsibility: loading FreshWiki data
    for evaluation purposes. It does NOT provide content for generation -
    that's the job of the SearchEngine and knowledge sources.
    
    The separation of concerns is crucial:
    - Knowledge sources (Wikipedia, web search) → Content generation
    - Evaluation datasets (FreshWiki) → Quality measurement
    """
    
    def __init__(self, data_path: str = "data/freshwiki"):
        self.data_path = Path(data_path)
        self.entries: List[FreshWikiEntry] = []
        self.topic_index: Dict[str, FreshWikiEntry] = {}
        
        # Load dataset on initialization
        self._load_evaluation_dataset()
    
    def _load_evaluation_dataset(self) -> None:
        """
        Load FreshWiki evaluation dataset from JSON files.
        
        This method looks for common FreshWiki file patterns and loads
        them into a standardized format for evaluation use.
        """
        if not self.data_path.exists():
            logger.warning(f"FreshWiki evaluation data not found at: {self.data_path}")
            logger.info("System will run without evaluation benchmarks")
            return
        
        # Common FreshWiki file patterns
        evaluation_files = [
            "freshwiki.json",
            "freshwiki_data.json", 
            "evaluation_set.json",
            "test.json",
            "validation.json"
        ]
        
        for filename in evaluation_files:
            file_path = self.data_path / filename
            if file_path.exists():
                logger.info(f"Loading FreshWiki evaluation data from: {file_path}")
                self._load_from_file(file_path)
                return
        
        # If no standard files found, try any JSON file
        json_files = list(self.data_path.glob("*.json"))
        if json_files:
            logger.info(f"Loading FreshWiki data from: {json_files[0]}")
            self._load_from_file(json_files[0])
        else:
            logger.warning("No FreshWiki evaluation files found - evaluation will be limited")
    
    def _load_from_file(self, file_path: Path) -> None:
        """Load and parse FreshWiki data from a specific JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different data structures
            if isinstance(data, list):
                # List of evaluation entries
                for entry_data in data:
                    entry = self._parse_evaluation_entry(entry_data)
                    if entry and entry.topic:  # Only add valid entries with topics
                        self.entries.append(entry)
                        self.topic_index[entry.topic.lower()] = entry
            
            elif isinstance(data, dict):
                if 'topic' in data or 'title' in data:
                    # Single evaluation entry
                    entry = self._parse_evaluation_entry(data)
                    if entry and entry.topic:
                        self.entries.append(entry)
                        self.topic_index[entry.topic.lower()] = entry
                else:
                    # Dictionary of entries
                    for key, entry_data in data.items():
                        entry = self._parse_evaluation_entry(entry_data)
                        if entry and entry.topic:
                            self.entries.append(entry)
                            self.topic_index[entry.topic.lower()] = entry
            
            logger.info(f"Loaded {len(self.entries)} evaluation entries from FreshWiki")
            
        except Exception as e:
            logger.error(f"Failed to load FreshWiki evaluation data from {file_path}: {e}")
    
    def _parse_evaluation_entry(self, data: Dict[str, Any]) -> Optional[FreshWikiEntry]:
        """
        Parse a single evaluation entry from various possible formats.
        
        This method is tolerant of different field naming conventions
        while extracting the essential components needed for evaluation.
        """
        try:
            # Extract topic/title (required)
            topic = (data.get('topic') or 
                    data.get('title') or 
                    data.get('article_title') or 
                    data.get('name', ''))
            
            if not topic:
                return None  # Skip entries without topics
            
            # Extract reference outline/headings for structure evaluation
            outline = []
            if 'outline' in data:
                outline = data['outline'] if isinstance(data['outline'], list) else []
            elif 'headings' in data:
                outline = data['headings'] if isinstance(data['headings'], list) else []
            elif 'sections' in data and isinstance(data['sections'], list):
                outline = data['sections']
            
            # Extract reference content for content evaluation
            content = (data.get('content') or 
                      data.get('text') or 
                      data.get('article_content') or 
                      data.get('full_text', ''))
            
            # Extract section-wise content for detailed evaluation
            sections = {}
            if 'sections' in data and isinstance(data['sections'], dict):
                sections = data['sections']
            elif 'section_content' in data:
                sections = data['section_content']
            
            # Extract metadata
            metadata = data.get('metadata', {})
            
            return FreshWikiEntry(
                topic=topic,
                reference_outline=outline,
                reference_content=content,
                ground_truth_sections=sections,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse FreshWiki evaluation entry: {e}")
            return None
    
    def get_evaluation_entry(self, topic: str) -> Optional[FreshWikiEntry]:
        """
        Get evaluation ground truth for a specific topic.
        
        This is the primary method used by the evaluation system to get
        reference data for comparing against generated content.
        """
        return self.topic_index.get(topic.lower())
    
    def get_all_evaluation_topics(self) -> List[str]:
        """Get list of all topics available for evaluation."""
        return [entry.topic for entry in self.entries]
    
    def get_evaluation_sample(self, n: int = 5) -> List[FreshWikiEntry]:
        """
        Get a random sample of evaluation entries for benchmarking.
        
        This is useful for running quick evaluation tests or when you want
        to benchmark on a subset of the full evaluation dataset.
        """
        import random
        return random.sample(self.entries, min(n, len(self.entries)))
    
    def validate_evaluation_dataset(self) -> Dict[str, Any]:
        """
        Validate the evaluation dataset and return comprehensive statistics.
        
        This helps you understand the quality and coverage of your evaluation
        data, which is crucial for meaningful benchmarking.
        """
        if not self.entries:
            return {
                'status': 'no_data',
                'message': 'No FreshWiki evaluation data loaded',
                'recommendation': 'Add FreshWiki JSON files to data/freshwiki/ directory'
            }
        
        stats = {
            'status': 'loaded',
            'total_entries': len(self.entries),
            'topics_with_outlines': 0,
            'topics_with_content': 0,
            'topics_with_sections': 0,
            'average_outline_length': 0,
            'average_content_length': 0,
            'content_length_distribution': {'short': 0, 'medium': 0, 'long': 0},
            'sample_topics': []
        }
        
        outline_lengths = []
        content_lengths = []
        
        for entry in self.entries:
            # Count entries with different types of ground truth data
            if entry.reference_outline:
                stats['topics_with_outlines'] += 1
                outline_lengths.append(len(entry.reference_outline))
            
            if entry.reference_content:
                stats['topics_with_content'] += 1
                content_length = len(entry.reference_content)
                content_lengths.append(content_length)
                
                # Categorize content length for distribution analysis
                if content_length < 1000:
                    stats['content_length_distribution']['short'] += 1
                elif content_length < 3000:
                    stats['content_length_distribution']['medium'] += 1
                else:
                    stats['content_length_distribution']['long'] += 1
            
            if entry.ground_truth_sections:
                stats['topics_with_sections'] += 1
            
            # Collect sample topics for manual inspection
            if len(stats['sample_topics']) < 5:
                stats['sample_topics'].append(entry.topic)
        
        # Calculate averages
        if outline_lengths:
            stats['average_outline_length'] = sum(outline_lengths) / len(outline_lengths)
        
        if content_lengths:
            stats['average_content_length'] = sum(content_lengths) / len(content_lengths)
        
        # Add data quality assessment
        stats['data_quality'] = {
            'outline_coverage': stats['topics_with_outlines'] / stats['total_entries'],
            'content_coverage': stats['topics_with_content'] / stats['total_entries'],
            'section_coverage': stats['topics_with_sections'] / stats['total_entries']
        }
        
        return stats


