from dataclasses import dataclass
from typing import List, Dict, Any
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import requests

from configs.base_config import RunnerArgument
from configs.writer_config import WriterConfig


class WriterAgent:
    """Single-agent writer pipeline"""
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

        # Initialize embedding model and FAISS index for retrieval
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(args.corpus_embeddings_path)
        # Load mapping from vector IDs to document text
        idx_map = args.corpus_embeddings_path.replace(".index", "_texts.pkl")
        with open(idx_map, "rb") as f:
            self.doc_texts: List[str] = pickle.load(f)

        # State placeholders
        self.knowledge_base: List[Dict[str, Any]] = []
        self.outline: Dict[str, Any] = {}
        self.draft_content: str = "" 

    def _call_api(self, prompt: str) -> str:
        """Send prompt to HF Inference API and return generated text"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
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
        """Semantic search against FAISS for initial topic grounding"""
        q_emb = self.embedder.encode(self.topic)
        faiss.normalize_L2(q_emb.reshape(1, -1))
        distances, indices = self.index.search(q_emb.reshape(1, -1), self.args.retrieve_top_k)
        self.knowledge_base = [{"content": self.doc_texts[idx], "source_id": idx} for idx in indices[0]]
        return self.knowledge_base

    def generate_outline(self) -> Dict[str, Any]:
        """One-shot API call to produce JSON outline"""
        snippets = "\n".join(r['content'] for r in self.knowledge_base)
        prompt = (
            f"You are an expert writer. Using these research snippets:\n{snippets}\n"
            f"Produce a JSON outline for '{self.topic}' with 3–5 sections, each having 'title' and optional 'key_points'."
        )
        out = self._call_api(prompt)
        import json
        try:
            self.outline = json.loads(out)
        except json.JSONDecodeError:
            start, end = out.find('{'), out.rfind('}')
            self.outline = json.loads(out[start:end+1])
        return self.outline

    def create_draft(self) -> str:  
        """Draft each section based on the outline via API calls"""
        parts = [f"# {self.topic}\n"]
        for sec in self.outline.get('sections', []):
            title = sec.get('title')
            kps = sec.get('key_points', [])
            kp_text = "\n".join(f"- {pt}" for pt in kps)
            prompt = (
                f"Write a ~200-word section titled '{title}' using these points:\n{kp_text}\n"
            )
            text = self._call_api(prompt)
            parts.append(f"## {title}\n{text}\n")
        self.draft_content = "\n".join(parts) 
        return self.draft_content

    def edit(self) -> str:
        """Polish the full draft via API call"""
        prompt = (
            f"Improve clarity, fix grammar, and enhance style for the following content:\n{self.draft_content}" 
        )
        polished = self._call_api(prompt)
        return polished

    def run_pipeline(self) -> str:
        if self.args.research_phase_enabled:
            self.research()
        if self.args.outline_generation_enabled:
            self.generate_outline()
        self.create_draft()  
        return self.edit()