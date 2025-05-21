import os


class WriterConfig:
    def __init__(
        self,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        retrieve_top_k: int = 10,
        knowledge_organization_enabled: bool = True,
        section_drafting_style: str = "informative",  # Could be "informative", "narrative", etc.
    ):
        self.hf_token = os.getenv("HF_TOKEN")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.retrieve_top_k = retrieve_top_k
        self.knowledge_organization_enabled = knowledge_organization_enabled
        self.section_drafting_style = section_drafting_style
