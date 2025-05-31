# src/evaluation/benchmarks/freshwiki_loader.py - Updated for actual FreshWiki format
import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import re

logger = logging.getLogger(__name__)

@dataclass
class FreshWikiEntry:
    """
    Represents a FreshWiki evaluation benchmark entry.
    """
    topic: str                              # The topic/title we're generating content for
    reference_outline: List[str]            # Ground truth section headings for structure evaluation
    reference_content: str                  # Full reference article for content comparison
    ground_truth_sections: Dict[str, str]   # Section-by-section ground truth for detailed analysis
    metadata: Dict[str, Any]               # Additional metadata (dates, categories, etc.)
    
    @classmethod
    def from_freshwiki_files(cls, json_file: Path, txt_file: Path) -> 'FreshWikiEntry':
        """
        Create FreshWikiEntry from FreshWiki JSON and TXT files.
        """
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Load TXT content
        with open(txt_file, 'r', encoding='utf-8') as f:
            txt_content = f.read()
        
        # Extract topic from title
        topic = json_data.get('title', '').replace('_', ' ')
        
        # Extract outline from JSON structure
        outline = cls._extract_outline_from_json(json_data)
        
        # Extract sections from TXT content
        sections = cls._extract_sections_from_txt(txt_content)
        
        return cls(
            topic=topic,
            reference_outline=outline,
            reference_content=txt_content,
            ground_truth_sections=sections,
            metadata={
                'url': json_data.get('url', ''),
                'summary': json_data.get('summary', ''),
                'source': 'freshwiki'
            }
        )
    
    @classmethod
    def _extract_outline_from_json(cls, json_data: Dict) -> List[str]:
        """Extract section headings from JSON content structure."""
        outline = []
        
        def extract_sections(content_list, level=0):
            for item in content_list:
                if isinstance(item, dict):
                    section_title = item.get('section_title', '').strip()
                    if section_title and section_title not in outline:
                        outline.append(section_title)
                    
                    # Recursively process subsections
                    subsections = item.get('subsections', [])
                    if subsections:
                        extract_sections(subsections, level + 1)
        
        content = json_data.get('content', [])
        if isinstance(content, list):
            extract_sections(content)
        
        return outline
    
    @classmethod
    def _extract_sections_from_txt(cls, txt_content: str) -> Dict[str, str]:
        """Extract sections from TXT content using markdown-style headers."""
        sections = {}
        lines = txt_content.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            
            # Check for headers (# or ##)
            if line.startswith('#'):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.lstrip('#').strip()
                current_content = []
            elif current_section:
                # Add content to current section
                current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections

class FreshWikiLoader:
    """
    Loader for FreshWiki evaluation dataset with JSON/TXT file pairs.
    """
    
    def __init__(self, data_path: str = "data/freshwiki"):
        self.data_path = Path(data_path)
        self.entries: List[FreshWikiEntry] = []
        self.topic_index: Dict[str, FreshWikiEntry] = {}
        
        # Load dataset on initialization
        self._load_evaluation_dataset()
    
    def _load_evaluation_dataset(self) -> None:
        """
        Load FreshWiki evaluation dataset from JSON and TXT subdirectories.
        """
        if not self.data_path.exists():
            logger.warning(f"FreshWiki evaluation data not found at: {self.data_path}")
            return
        
        json_dir = self.data_path / "json"
        txt_dir = self.data_path / "txt"
        
        if not json_dir.exists() or not txt_dir.exists():
            logger.warning(f"FreshWiki subdirectories not found. Expected: {json_dir} and {txt_dir}")
            return
        
        # Find matching JSON and TXT files
        json_files = {f.stem: f for f in json_dir.glob("*.json")}
        txt_files = {f.stem: f for f in txt_dir.glob("*.txt")}
        
        # Load entries for files that have both JSON and TXT versions
        common_stems = set(json_files.keys()) & set(txt_files.keys())
        
        logger.info(f"Found {len(common_stems)} FreshWiki entries with both JSON and TXT files")
        
        for stem in common_stems:
            try:
                entry = FreshWikiEntry.from_freshwiki_files(
                    json_files[stem], 
                    txt_files[stem]
                )
                
                if entry.topic:
                    self.entries.append(entry)
                    self.topic_index[entry.topic.lower()] = entry
                    
            except Exception as e:
                logger.warning(f"Failed to load FreshWiki entry {stem}: {e}")
        
        logger.info(f"Successfully loaded {len(self.entries)} FreshWiki evaluation entries")
    
    def get_evaluation_entry(self, topic: str) -> Optional[FreshWikiEntry]:
        """Get evaluation ground truth for a specific topic."""
        return self.topic_index.get(topic.lower())
    
    def get_all_evaluation_topics(self) -> List[str]:
        """Get list of all topics available for evaluation."""
        return [entry.topic for entry in self.entries]
    
    def get_evaluation_sample(self, n: int = 5) -> List[FreshWikiEntry]:
        """Get a random sample of evaluation entries for benchmarking."""
        import random
        return random.sample(self.entries, min(n, len(self.entries)))
    
    def validate_evaluation_dataset(self) -> Dict[str, Any]:
        """Validate the evaluation dataset and return comprehensive statistics."""
        if not self.entries:
            return {
                'status': 'no_data',
                'message': 'No FreshWiki evaluation data loaded',
                'recommendation': 'Ensure JSON and TXT subdirectories exist in data/freshwiki/'
            }
        
        stats = {
            'status': 'loaded',
            'total_entries': len(self.entries),
            'topics_with_outlines': 0,
            'topics_with_content': 0,
            'topics_with_sections': 0,
            'average_outline_length': 0,
            'average_content_length': 0,
            'sample_topics': []
        }
        
        outline_lengths = []
        content_lengths = []
        
        for entry in self.entries:
            if entry.reference_outline:
                stats['topics_with_outlines'] += 1
                outline_lengths.append(len(entry.reference_outline))
            
            if entry.reference_content:
                stats['topics_with_content'] += 1
                content_lengths.append(len(entry.reference_content))
            
            if entry.ground_truth_sections:
                stats['topics_with_sections'] += 1
            
            if len(stats['sample_topics']) < 5:
                stats['sample_topics'].append(entry.topic)
        
        # Calculate averages
        if outline_lengths:
            stats['average_outline_length'] = sum(outline_lengths) / len(outline_lengths)
        
        if content_lengths:
            stats['average_content_length'] = sum(content_lengths) / len(content_lengths)
        
        return stats