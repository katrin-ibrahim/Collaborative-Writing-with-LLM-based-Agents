import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from utils.ollama_client import OllamaClient
from config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class SelfRAGRunner:
    """
    Runner for Self-RAG using the official implementation.
    
    This integrates the official Self-RAG code with Ollama models.
    """
    
    def __init__(self, ollama_client: OllamaClient, model_config: ModelConfig,
                 self_rag_path: Optional[str] = None):
        
        print(self_rag_path)
        self.client = ollama_client
        self.model_config = model_config
        
        # Path to integrated Self-RAG repository
        self.self_rag_path = self_rag_path or os.path.join(os.getcwd(), "self_rag")
        print(self.self_rag_path)
        
        # Check if Self-RAG is available
        if not self._check_self_rag_installation():
            logger.error("Self-RAG not found in expected location.")
            logger.error(f"Expected path: {self.self_rag_path}")
            raise RuntimeError("Self-RAG integration not available")
    
    def _check_self_rag_installation(self) -> bool:
        """Check if Self-RAG repository is available."""
        return os.path.exists(self.self_rag_path) and \
               os.path.exists(os.path.join(self.self_rag_path, "retrieval_lm"))
    
    def setup_self_rag(self):
        """Setup Self-RAG environment (one-time setup)."""
        if not self._check_self_rag_installation():
            logger.info("Cloning Self-RAG repository...")
            subprocess.run([
                "git", "clone", "https://github.com/AkariAsai/self-rag.git",
                self.self_rag_path
            ], check=True)
        
        # Install requirements if needed
        requirements_file = os.path.join(self.self_rag_path, "requirements.txt")
        if os.path.exists(requirements_file):
            logger.info("Installing Self-RAG requirements...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ], check=True)
    
    def generate_with_self_rag(self, topic: str) -> str:
        """
        Generate article using Self-RAG approach with Ollama.
        
        Since the official Self-RAG uses specific models, we'll implement
        the core Self-RAG algorithm using Ollama models.
        """
        logger.info(f"Generating with Self-RAG for: {topic}")
        
        try:
            # Self-RAG Algorithm:
            # 1. Determine if retrieval is needed
            # 2. If yes, retrieve relevant passages
            # 3. Generate with retrieved context
            # 4. Self-reflect on the generation
            # 5. Refine if necessary
            
            # Step 1: Retrieval decision
            need_retrieval = self._decide_retrieval(topic)
            
            # Step 2: Retrieve if needed
            retrieved_passages = []
            if need_retrieval:
                retrieved_passages = self._retrieve_passages(topic)
            
            # Step 3: Initial generation
            initial_content = self._generate_with_context(topic, retrieved_passages)
            
            # Step 4: Self-reflection
            reflection = self._self_reflect(topic, initial_content, retrieved_passages)
            
            # Step 5: Refinement based on reflection
            if reflection["needs_refinement"]:
                final_content = self._refine_generation(
                    topic, initial_content, reflection, retrieved_passages
                )
            else:
                final_content = initial_content
            
            # Format as article
            if not final_content.startswith("#"):
                final_content = f"# {topic}\n\n{final_content}"
            
            return final_content
            
        except Exception as e:
            logger.error(f"Self-RAG generation failed: {e}")
            return f"# {topic}\n\nSelf-RAG generation failed: {str(e)}"
    
    def _decide_retrieval(self, topic: str) -> bool:
        """Decide if retrieval is needed for the topic."""
        prompt = f"""Determine if external retrieval is needed to write about "{topic}".

Consider:
1. Is this a factual topic requiring specific information?
2. Would retrieval improve accuracy and detail?
3. Is this a topic with recent developments?

Respond with only YES or NO."""

        response = self.client.call_api(
            prompt=prompt,
            model=self.model_config.retrieval_model,
            temperature=0.3,
            max_tokens=10
        )
        
        return "yes" in response.lower()
    
    def _retrieve_passages(self, topic: str, num_passages: int = 5) -> List[Dict[str, str]]:
        """Retrieve relevant passages for the topic."""
        # Generate search queries
        search_prompt = f"""Generate {num_passages} diverse search queries for researching "{topic}".
Each query should explore a different aspect of the topic.

Format: One query per line, no numbers or bullets."""

        queries_response = self.client.call_api(
            prompt=search_prompt,
            model=self.model_config.retrieval_model,
            temperature=0.7,
            max_tokens=200
        )
        
        queries = [q.strip() for q in queries_response.split('\n') if q.strip()][:num_passages]
        
        # Simulate retrieval (in production, this would use actual retrieval)
        # For now, we'll use mock passages
        passages = []
        for i, query in enumerate(queries):
            passages.append({
                "query": query,
                "content": f"Retrieved information about {topic} related to: {query}. This passage contains relevant facts and details that help in writing a comprehensive article.",
                "relevance": 0.8 - (i * 0.1)
            })
        
        return passages
    
    def _generate_with_context(self, topic: str, passages: List[Dict[str, str]]) -> str:
        """Generate content with retrieved context."""
        if passages:
            context = "\n\n".join([
                f"[Passage {i+1}]: {p['content']}" 
                for i, p in enumerate(passages)
            ])
            
            prompt = f"""Write a comprehensive article about "{topic}" using the following retrieved information:

{context}

Requirements:
1. Use the retrieved information to support your writing
2. Create a well-structured article with multiple sections
3. Include specific details from the passages
4. Maintain factual accuracy based on the provided context
5. Use markdown formatting

Article:"""
        else:
            prompt = f"""Write a comprehensive article about "{topic}".

Requirements:
1. Create a well-structured article with multiple sections
2. Use your knowledge to provide accurate information
3. Include introduction, main sections, and conclusion
4. Use markdown formatting

Article:"""

        content = self.client.call_api(
            prompt=prompt,
            model=self.model_config.generation_model,
            temperature=0.7,
            max_tokens=1500
        )
        
        return content
    
    def _self_reflect(self, topic: str, content: str, passages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Self-reflect on the generated content."""
        reflection_prompt = f"""Evaluate this article about "{topic}":

{content}

Evaluate based on:
1. Factual accuracy and completeness
2. Structure and organization
3. Use of retrieved information (if any)
4. Coverage of important aspects
5. Overall quality

Provide a brief analysis and indicate if refinement is needed.
End with: NEEDS_REFINEMENT: YES or NO"""

        reflection_response = self.client.call_api(
            prompt=reflection_prompt,
            model=self.model_config.reflection_model,
            temperature=0.3,
            max_tokens=300
        )
        
        needs_refinement = "needs_refinement: yes" in reflection_response.lower()
        
        return {
            "needs_refinement": needs_refinement,
            "feedback": reflection_response,
            "issues": self._extract_issues(reflection_response)
        }
    
    def _extract_issues(self, reflection: str) -> List[str]:
        """Extract specific issues from reflection."""
        issues = []
        
        issue_keywords = [
            "missing", "lacks", "insufficient", "incorrect", 
            "unclear", "poorly", "needs more", "should include"
        ]
        
        lines = reflection.lower().split('.')
        for line in lines:
            if any(keyword in line for keyword in issue_keywords):
                issues.append(line.strip())
        
        return issues[:3]  # Top 3 issues
    
    def _refine_generation(self, topic: str, content: str, 
                          reflection: Dict[str, Any], passages: List[Dict[str, str]]) -> str:
        """Refine the generation based on self-reflection."""
        issues_text = "\n".join([f"- {issue}" for issue in reflection["issues"]])
        
        refinement_prompt = f"""Improve this article about "{topic}" based on the following feedback:

Current article:
{content}

Issues identified:
{issues_text}

Full feedback:
{reflection['feedback']}

Please rewrite the article addressing these issues while maintaining all good aspects.

Improved article:"""

        refined_content = self.client.call_api(
            prompt=refinement_prompt,
            model=self.model_config.generation_model,
            temperature=0.6,
            max_tokens=1500
        )
        
        return refined_content


class SelfRAGOfficialWrapper:
    """
    Alternative wrapper that directly uses the official Self-RAG implementation.
    This requires the official model weights and setup.
    """
    
    def __init__(self, self_rag_path: str):
        self.self_rag_path = Path(self_rag_path)
        sys.path.insert(0, str(self.self_rag_path))
        
        try:
            from retrieval_lm.run_short_form import run_generation
            self.run_generation = run_generation
            logger.info("Official Self-RAG loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import Self-RAG: {e}")
            self.run_generation = None
    
    def generate(self, topic: str, use_retrieval: bool = True) -> str:
        """Generate using official Self-RAG implementation."""
        if self.run_generation is None:
            return f"# {topic}\n\nOfficial Self-RAG not available"
        
        # Configure Self-RAG parameters
        args = {
            "input_query": f"Write a comprehensive article about {topic}",
            "model_name": "selfrag/selfrag_llama2_7b",
            "use_retrieval": use_retrieval,
            "ndocs": 5,
            "max_new_tokens": 1000,
            "threshold": 0.2
        }
        
        try:
            # Run official Self-RAG
            output = self.run_generation(**args)
            return f"# {topic}\n\n{output}"
        except Exception as e:
            logger.error(f"Official Self-RAG failed: {e}")
            return f"# {topic}\n\nError: {str(e)}"