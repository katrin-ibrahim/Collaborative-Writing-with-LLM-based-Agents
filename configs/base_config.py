class RunnerArgument:
    def __init__(
        self,
        topic: str,
        corpus_embeddings_path: str,
        retrieve_top_k: int = 10,
        research_phase_enabled: bool = True,
        outline_generation_enabled: bool = True,
        brainstorming_enabled: bool = True
    ):
        self.topic = topic
        self.corpus_embeddings_path = corpus_embeddings_path
        self.retrieve_top_k = retrieve_top_k
        self.research_phase_enabled = research_phase_enabled
        self.outline_generation_enabled = outline_generation_enabled
        self.brainstorming_enabled = brainstorming_enabled