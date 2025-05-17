from agents.writer import WriterAgent
from configs.base_config import RunnerArgument
from configs.writer_config import WriterConfig
from utils.data_loader import load_wildseek
import os


def main():
    topics = load_wildseek()
  
    config = WriterConfig()
    args = RunnerArgument(
        topic="Deep-Sea Vents",
        corpus_embeddings_path="data/faiss/wildseek.index"
    )

    agent = WriterAgent(config, args)
    agent.doc_texts = topics  

    final_content = agent.run_pipeline()
    print("=== Final Content ===")
    print(final_content)


if __name__ == "__main__":
    main()

    



