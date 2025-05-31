#!/usr/bin/env python3
"""
RAG-Writer baseline experiment script.
This baseline uses a retrieval-augmented generation approach with a single writer agent.
"""

import sys
import os
import argparse
from pathlib import Path

# Add the src directory to the path so we can import from there
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src"))

from workflows.rag_writer import RAGWriterWorkflow
from evaluation.evaluator import Evaluator
from utils.config import load_config


def main():
    """Run the RAG-writer baseline experiment."""
    parser = argparse.ArgumentParser(description="Run RAG-writer baseline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/baseline_configs/rag_writer.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="experiments/results/rag_writer",
        help="Directory to save results"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the workflow
    workflow = RAGWriterWorkflow(config)
    
    # Initialize the evaluator
    evaluator = Evaluator(config)
    
    # Run experiments
    print("Running RAG-writer baseline experiments...")
    results = workflow.run()
    
    # Evaluate results
    print("Evaluating results...")
    evaluation_results = evaluator.evaluate(results)
    
    # Save results
    print(f"Saving results to {args.output_dir}")
    evaluator.save_results(evaluation_results, args.output_dir)
    
    print("RAG-writer baseline experiments complete!")


if __name__ == "__main__":
    main()
