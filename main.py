import os
import json
from datetime import datetime  # Changed from import datetime
from agents.writer import WriterAgent
from configs.base_config import RunnerArgument
from configs.writer_config import WriterConfig
from utils.data_loader import load_wildseek
from evaluation.content_evaluator import ContentEvaluator

def main():
    # Load WildSeek topics for knowledge base
    print("=== Loading WildSeek Topics ===")
    topics = load_wildseek()
  
    # Configure the writer agent
    config = WriterConfig()
    args = RunnerArgument(
        topic=topics[0],  # Use the first topic for demonstration
        corpus_embeddings_path=""
    )

    # Initialize the writer agent (baseline approach)
    agent = WriterAgent(config, args)

    # Run the baseline non-iterative pipeline
    final_content = agent.run_pipeline()
    
    # Save the final content to a file
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(final_content)
        
    # # Evaluate the content
    # print("\n=== Evaluating Content ===")
    # evaluator = ContentEvaluator(api_key=os.getenv("HF_TOKEN"))
    # evaluation = evaluator.evaluate_article(final_content, args.topic)
    
    # # Print key metrics
    # print(f"Overall Score: {evaluation['overall_score']}/10")
    # print(f"Word Count: {evaluation['metrics']['word_count']}")
    # print(f"Sections: {evaluation['metrics']['section_count']}")
    
    # if 'llm_evaluation' in evaluation and 'summary' in evaluation['llm_evaluation']:
    #     print(f"\nSummary: {evaluation['llm_evaluation']['summary']}")
        
    # # Save the results with timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # results = {
    #     "topic": args.topic,
    #     "timestamp": timestamp,
    #     "content": final_content,
    #     "evaluation": evaluation
    # }
    
    # os.makedirs("results", exist_ok=True)
    # with open(f"results/{args.topic.replace(' ', '_')}_{timestamp}.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()