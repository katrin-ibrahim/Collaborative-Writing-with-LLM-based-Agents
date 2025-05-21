from dataclasses import dataclass
import logging
import os
import re
import json
from typing import List, Dict, Any, Optional
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import requests

from configs.base_config import RunnerArgument
from configs.writer_config import WriterConfig
from knowledge.knowledge_base import KnowledgeBase, KnowledgeNode, WikipediaRetriever

def _extract_json(self, text):
    """Extract JSON from text that might have explanatory content around it"""
    # Find anything between { and }
    json_pattern = r'({[\s\S]*})'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    return None


class WriterAgent:
    """Writer agent following CO-STORM framework using open-source LLMs"""
    def __init__(
        self,
        config: WriterConfig,
        args: RunnerArgument
    ):
        self.topic = args.topic
        self.config = config
        self.args = args

              # Prepare HF Inference endpoint URL and headers
        self.api_url = f"https://router.huggingface.co/novita/v3/openai/chat/completions"
        self.headers = {"Authorization": f"Bearer {config.hf_token}"} if config.hf_token else {}
        logging.info(f"Using HF Inference API with token: {config.hf_token is not None}")

        # Initialize embedding model for semantic similarity in knowledge organization
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # State placeholders similar to CO-STORM
        self.knowledge_base = KnowledgeBase(topic=self.topic)
        self.wiki_retriever = WikipediaRetriever()  # Initialize the Wikipedia retriever
        self.conversation_history = []
        self.draft_content = ""

    def _call_api(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send prompt to HF Inference API and return generated text"""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system", 
                "content": system_prompt
            })
            
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        payload = {
            "messages": messages,
            "model": "deepseek/deepseek-v3-0324",
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        output = response.json()
        
        # Handle OpenAI-compatible response format
        if isinstance(output, dict) and "choices" in output:
            choices = output.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                if isinstance(message, dict) and "content" in message:
                    return message["content"].strip()
        
        # Handle older Hugging Face formats
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"].strip()
        if isinstance(output, dict) and "generated_text" in output:
            return output["generated_text"].strip()
        
        raise RuntimeError(f"Unexpected API response: {output}")

    def research(self) -> List[Dict[str, Any]]:
        """Get knowledge from Wikipedia"""
        print("Gathering knowledge from Wikipedia...")
        
        # Retrieve Wikipedia content using the wiki_retriever
        retrieve_count = self.args.retrieve_top_k if hasattr(self.args, 'retrieve_top_k') else 3
        wiki_snippets = self.wiki_retriever.get_wiki_content(
            self.topic, 
            max_articles=retrieve_count
        )
        
        # Add to knowledge base root node
        for snippet in wiki_snippets:
            self.knowledge_base.root.add_snippet(snippet)
        
        return wiki_snippets

    def organize_knowledge(self) -> None:
        """knowledge organization into hierarchical structure"""
        print("Organizing knowledge into structured topics...")
        
        # Extract all snippets
        all_snippets = self.knowledge_base.root.snippets
        print(f"Total snippets collected: {len(all_snippets)}")

        snippets_text = "\n\n".join([f"Snippet {i+1}: {s['content']}" for i, s in enumerate(all_snippets)])
        print(f"Snippets text for LLM:\n{snippets_text[:500]}...")  # Print first 500 chars for debugging
        # Use LLM to identify main topics
        system_prompt = (
        "You are a knowledge organization assistant. Your response must be valid JSON only, "
        "with no additional explanations, markdown formatting or text outside the JSON structure."
        )
        prompt = (
            f"I have collected information about '{self.topic}' and need to organize it into a hierarchical structure.\n\n"
            f"Here are the information snippets:\n{snippets_text}\n\n"
            f"Please identify 3-5 main topics that would make good sections for an article about '{self.topic}'.\n"
            f"For each topic, list 2-3 subtopics if appropriate.\n\n"
            f"Return your response as a JSON with this structure:\n"
            f"{{\"topics\": [{{\n"
            f"  \"title\": \"Main Topic 1\",\n"
            f"  \"subtopics\": [\"Subtopic 1A\", \"Subtopic 1B\"]\n"
            f"}}, ...]}}"
        )
        
        response = self._call_api(prompt, system_prompt)
        
        # Parse response
        try:
            topics_data = _extract_json(response)
            if "topics" in topics_data:
                # Create new knowledge base structure
                new_root = KnowledgeNode(title=self.topic)
                
                for topic in topics_data["topics"]:
                    topic_node = KnowledgeNode(title=topic["title"])
                    new_root.children.append(topic_node)
                    
                    # Add subtopics
                    for subtopic in topic.get("subtopics", []):
                        subtopic_node = KnowledgeNode(title=subtopic)
                        topic_node.children.append(subtopic_node)
                
                # Assign snippets to appropriate nodes using similarity
                for snippet in all_snippets:
                    best_node = new_root
                    best_score = 0
                    
                    # Check each topic node
                    snippet_emb = self.embedder.encode(snippet["content"])
                    
                    for topic_node in new_root.children:
                        topic_emb = self.embedder.encode(topic_node.title)
                        score = snippet_emb @ topic_emb
                        
                        if score > best_score:
                            best_score = score
                            best_node = topic_node
                        
                        # Check subtopics
                        for subtopic_node in topic_node.children:
                            subtopic_emb = self.embedder.encode(subtopic_node.title)
                            subscore = snippet_emb @ subtopic_emb
                            
                            if subscore > best_score:
                                best_score = subscore
                                best_node = subtopic_node
                    
                    # Assign to best matching node
                    best_node.add_snippet(snippet)
                
                # Update knowledge base root
                self.knowledge_base.root = new_root
        except json.JSONDecodeError:
            print("Failed to parse knowledge organization response. Keeping flat structure.")
            self._generate_fallback_structure(all_snippets)
    
    def create_outline(self) -> Dict[str, Any]:
        """Generate structured outline from knowledge base similar to CO-STORM"""
        print("Creating detailed outline...")
        
        # Generate outline from knowledge base structure
        outline = self.knowledge_base.to_outline()
        
        # For each section without key points, generate them
        for section in outline["sections"]:
            if not section.get("key_points") or len(section["key_points"]) < 3:
                prompt = (
                    f"For an article section titled '{section['title']}' about '{self.topic}', "
                    f"generate 3-5 key points that should be covered. "
                    f"Each key point should be a single sentence or phrase. "
                    f"Return the key points as a JSON array of strings."
                )
                response = self._call_api(prompt)
                
                try:
                    key_points = json.loads(response)
                    if isinstance(key_points, list):
                        section["key_points"] = key_points
                    elif isinstance(key_points, dict) and "key_points" in key_points:
                        section["key_points"] = key_points["key_points"]
                except json.JSONDecodeError:
                    # Try to extract a list from text
                    points = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', response, re.DOTALL)
                    if points:
                        section["key_points"] = [p.strip() for p in points]
                    else:
                        # Fallback to splitting by newlines
                        section["key_points"] = [
                            line.strip().strip('- ').strip('* ').strip()
                            for line in response.split('\n')
                            if line.strip() and not line.strip().startswith('#')
                        ]
        
        return outline
    
    def draft_section(self, section: Dict[str, Any]) -> str:
        """Draft a single section using LLM"""
        title = section["title"]
        key_points = section.get("key_points", [])
        
        # Find relevant snippets in knowledge base
        relevant_snippets = []
        if not hasattr(self.knowledge_base, 'root') or not self.knowledge_base.root.children:
            print("No knowledge base structure found. Using flat snippets.")
            relevant_snippets = self.knowledge_base.root.snippets
        for node in self.knowledge_base.root.children:
            if title.lower() in node.title.lower() or node.title.lower() in title.lower():
                relevant_snippets.extend(node.snippets)
            
            for child in node.children:
                if title.lower() in child.title.lower() or child.title.lower() in title.lower():
                    relevant_snippets.extend(child.snippets)
        
        # If we have snippets, include them in the context
        snippets_text = ""
        if relevant_snippets:
            snippets_text = "Reference snippets:\n" + "\n\n".join(
                [s["content"] for s in relevant_snippets[:3]]
            )
        
        # Format key points
        key_points_text = "\n".join(f"- {point}" for point in key_points)
        
        system_prompt = (
            f"You are writing a section for an informative article about {self.topic}. "
            f"Write in a clear, engaging style with well-structured paragraphs."
        )
        
        prompt = (
            f"Write a detailed section titled '{title}' for an article about '{self.topic}'.\n\n"
            f"Cover these key points:\n{key_points_text}\n\n"
            f"{snippets_text}\n\n"
            f"Write approximately 300-500 words with clear topic sentences, supporting evidence, "
            f"and smooth transitions between ideas. Use engaging language appropriate for an informative article."
        )
        
        return self._call_api(prompt, system_prompt)
    
    def create_draft(self, outline: Dict[str, Any]) -> str:
        """CO-STORM style drafting based on outline and knowledge base"""
        print("Creating full draft from outline...")
        sections = outline.get("sections", [])
        
        parts = [f"# {self.topic}\n\n"]
        
        # Introduction
        intro_prompt = (
            f"Write an engaging introduction (about 150 words) for an article about '{self.topic}'. "
            f"The introduction should provide context, highlight importance, and briefly preview "
            f"the main points that will be covered in the article."
        )
        intro = self._call_api(intro_prompt)
        parts.append(f"{intro}\n\n")
        
        # Draft each section
        for section in sections:
            section_content = self.draft_section(section)
            parts.append(f"## {section['title']}\n\n{section_content}\n\n")
        
        # Conclusion
        conclusion_prompt = (
            f"Write a conclusion (about 150 words) for an article about '{self.topic}'. "
            f"The conclusion should summarize the main points, emphasize the significance "
            f"of the topic, and leave readers with a final thought."
        )
        conclusion = self._call_api(conclusion_prompt)
        parts.append(f"## Conclusion\n\n{conclusion}\n\n")
        
        self.draft_content = "\n".join(parts)
        return self.draft_content
    
    
    def run_pipeline(self) -> str:
        """CO-STORM inspired pipeline with knowledge organization"""
        # Research phase
        print("=== Research Phase ===")
        self.research()
        
        # Knowledge organization (similar to CO-STORM knowledge base)
        print("=== Knowledge Organization Phase ===")
        self.organize_knowledge()
        
        # Outline creation
        print("=== Outline Creation Phase ===")
        outline = self.create_outline()
        
        # Drafting phase
        print("=== Draft Creation Phase ===")
        return self.create_draft(outline)
        
