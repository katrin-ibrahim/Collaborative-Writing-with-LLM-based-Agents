import os
import yaml
from huggingface_hub import InferenceApi

class WriterAgent:
    def __init__(self, config: dict):
        # Load core settings
        self.model_name = config['model_name']
        self.hf_token = os.getenv('HF_TOKEN')
        self.max_tokens = config.get('max_tokens', 200)
        self.prompt_template = config['prompt_template']

        # Initialize HF inference client
        self.client = InferenceApi(
            repo_id=self.model_name,
            token=self.hf_token
        )

    def generate(self, input_text: str) -> str:
        # Fill template
        prompt = self.prompt_template.format(input=input_text)
        # Call model API without raw_response
        response = self.client(
            prompt,
            params={
                'max_new_tokens': self.max_tokens
            },
            raw_response=True
        )
        # The response is the generated text directly when not using raw_response
        return response