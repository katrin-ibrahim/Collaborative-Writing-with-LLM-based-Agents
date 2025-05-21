from dataclasses import dataclass
import wikipedia
import re
import os
import json
import hashlib
from typing import List, Dict, Any, Optional

class WikipediaRetriever:
    def __init__(self, cache_dir: str = "data/wiki_cache"):
        """Initialize Wikipedia retriever with caching"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # Set user agent for Wikipedia API
        wikipedia.set_rate_limiting(True)
        
    def get_wiki_content(self, topic: str, max_articles: int = 3, max_sections: int = 5) -> List[Dict[str, Any]]:
        """Retrieve Wikipedia content with caching"""
        # Create a cache key based on the topic
        cache_key = hashlib.md5(f"{topic}_{max_articles}_{max_sections}".encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Check if we have a cached result
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # If not in cache, retrieve from Wikipedia
        snippets = self._retrieve_from_wiki(topic, max_articles, max_sections)
        
        # Cache the result
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(snippets, f, ensure_ascii=False, indent=2)
        
        return snippets
    
    def _retrieve_from_wiki(self, topic: str, max_articles: int = 3, max_sections: int = 5) -> List[Dict[str, Any]]:
        """Retrieve content from Wikipedia using search instead of exact title matching"""
        try:
            search_variations = [
            topic,
            topic.replace("-", " "),  # Replace hyphens with spaces
            topic.replace(" ", "-"),  # Replace spaces with hyphens
            " ".join(topic.split()[:-1]) if len(topic.split()) > 1 else topic,  # Remove last word
            topic.split()[0] if len(topic.split()) > 1 else topic,  # Just first word
        ]
        
            all_search_results = []
            for variation in search_variations:
                results = wikipedia.search(variation, results=max_articles)
                all_search_results.extend(results)
                if results:  # If we found something, we can stop trying variations
                    break
                
            # Remove duplicates while preserving order
            search_results = []
            for result in all_search_results:
                if result not in search_results:
                    search_results.append(result)
            
            # Limit to max_articles
            search_results = search_results[:max_articles]
            if not search_results:
                print(f"Warning: Could not find any Wikipedia pages related to '{topic}'")
                return []
            
                
            snippets = []
            source_id = 0
            
            for i, page_title in enumerate(search_results):
                if i >= max_articles:
                    break
                    
                try:
                    # Get page content
                    page = wikipedia.page(page_title, auto_suggest=True)
                    
                    # Add page summary
                    snippets.append({
                        "title": page.title,
                        "section": "Summary",
                        "content": page.summary,
                        "url": page.url,
                        "source_id": source_id
                    })
                    source_id += 1
                    
                    # Get sections
                    section_count = 0
                    for section_title in page.sections:
                        if section_count >= max_sections:
                            break
                            
                        try:
                            content = page.section(section_title)
                            if content and len(content) > 100:  # Skip empty or very small sections
                                snippets.append({
                                    "title": page.title,
                                    "section": section_title,
                                    "content": content,
                                    "url": page.url,
                                    "source_id": source_id
                                })
                                section_count += 1
                                source_id += 1
                        except Exception as e:
                            print(f"Error extracting section {section_title}: {e}")
                            
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    if e.options and len(e.options) > 0:
                        try:
                            # Try the first option
                            page = wikipedia.page(e.options[0], auto_suggest=False)
                            snippets.append({
                                "title": page.title,
                                "section": "Summary",
                                "content": page.summary,
                                "url": page.url,
                                "source_id": source_id
                            })
                            source_id += 1
                        except:
                            print(f"Failed to get alternative page for '{page_title}'")
                except Exception as e:
                    print(f"Error retrieving page '{page_title}': {e}")
                    
            print(f"Retrieved {len(snippets)} snippets for topic '{topic}'")
            print(f"Snippets: {snippets}")
            return snippets#
        except Exception as e:
            print(f"Error in Wikipedia retrieval for topic '{topic}': {e}")
            return []

@dataclass
class KnowledgeNode:
    """A hierarchical node for knowledge organization"""
    title: str = ""
    content: str = ""
    source: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    children: List['KnowledgeNode'] = None
    snippets: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.children is None:
            self.children = []
        if self.snippets is None:
            self.snippets = []
    
    def add_snippet(self, snippet: Dict[str, Any]):
        """Add a snippet to this node"""
        self.snippets.append(snippet)
        # Also update content if empty
        if not self.content and 'content' in snippet:
            self.content = snippet['content']
    
    def add_child(self, child: 'KnowledgeNode'):
        """Add a child node"""
        self.children.append(child)
        return child

class KnowledgeBase:
    """Hierarchical knowledge base for storing and retrieving information"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.root = KnowledgeNode(title=topic)
        self.wiki_retriever = WikipediaRetriever()
        
    def populate_from_wikipedia(self, max_articles: int = 3, max_sections: int = 5):
        """Populate knowledge base with information from Wikipedia"""
        wiki_content = self.wiki_retriever.get_wiki_content(
            self.topic, 
            max_articles=max_articles,
            max_sections=max_sections
        )
        
        for snippet in wiki_content:
            self.root.add_snippet(snippet)
        
        return len(wiki_content)
    
    def to_outline(self) -> Dict[str, Any]:
        """Convert knowledge base to outline structure"""
        sections = []
        
        # Convert each child node to a section
        for child in self.root.children:
            section = {
                "title": child.title,
                "key_points": [],
                "subtopics": [subchild.title for subchild in child.children]
            }
            sections.append(section)
            
            # Extract key points from snippets if available
            if child.snippets:
                # Simple approach: take first few sentences from each snippet
                for snippet in child.snippets[:2]:
                    content = snippet.get('content', '')
                    sentences = re.split(r'(?<=[.!?])\s+', content)
                    for sentence in sentences[:2]:
                        if sentence and len(sentence) > 20:  # Only meaningful sentences
                            section["key_points"].append(sentence)
        
        return {
            "title": self.topic,
            "sections": sections
        }
    
    def add_node(self, content: str, source: str, confidence: float = 1.0, metadata: Optional[Dict[str, Any]] = None):
        """Add a new knowledge node to the base (flat structure for backward compatibility)"""
        node = KnowledgeNode(content=content, source=source, confidence=confidence, metadata=metadata)
        self.root.children.append(node)
        
    def get_all_content(self) -> str:
        """Get all content from knowledge nodes concatenated"""
        texts = []
        
        # Get content from root snippets
        for snippet in self.root.snippets:
            if 'content' in snippet:
                texts.append(snippet['content'])
        
        # Get content from children
        for child in self.root.children:
            if child.content:
                texts.append(child.content)
            for snippet in child.snippets:
                if 'content' in snippet:
                    texts.append(snippet['content'])
            
            # Get from grandchildren too
            for grandchild in child.children:
                if grandchild.content:
                    texts.append(grandchild.content)
                for snippet in grandchild.snippets:
                    if 'content' in snippet:
                        texts.append(snippet['content'])
        
        return "\n\n".join(texts)
    
    def get_sources(self) -> List[str]:
        """Get list of all sources"""
        sources = []
        
        # Get sources from root snippets
        for snippet in self.root.snippets:
            if 'title' in snippet and 'url' in snippet:
                sources.append(f"{snippet['title']} ({snippet['url']})")
        
        # Get sources from children
        for node in self.root.children:
            if node.source:
                sources.append(node.source)
            for snippet in node.snippets:
                if 'title' in snippet and 'url' in snippet:
                    sources.append(f"{snippet['title']} ({snippet['url']})")
        
        return sources
    
    def clear(self):
        """Clear all nodes from knowledge base"""
        self.root = KnowledgeNode(title=self.topic)