#!/bin/bash
# Generated experiment script for agentic methods: agent_exp
# Generated on: 2025-11-03 16:05:32
#

echo "Starting experiment 1: A2_balanced_writer--writer_only--wm-section--revm-pending-sections--rm-wiki_03-11_1605"
echo "Results will be saved to: results/ollama/A2_balanced_writer--writer_only--wm-section--revm-pending-sections--rm-wiki_03-11_1605"
python -m src.main -en A2_balanced_writer--writer_only--wm-section--revm-pending-sections--rm-wiki_03-11_1605 --backend ollama --methods writer_only --num_topics 20 --model_config balanced_writer --retrieval_manager wiki --writing_mode section
echo "Experiment 1 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/A2_balanced_writer--writer_only--wm-section--revm-pending-sections--rm-wiki_03-11_1605"
echo "Completed experiment 1 with evaluation"
echo "---"

echo "Starting experiment 2: A2_balanced_writer--writer_only--wm-full_article--revm-pending-sections--rm-wiki_03-11_1605"
echo "Results will be saved to: results/ollama/A2_balanced_writer--writer_only--wm-full_article--revm-pending-sections--rm-wiki_03-11_1605"
python -m src.main -en A2_balanced_writer--writer_only--wm-full_article--revm-pending-sections--rm-wiki_03-11_1605 --backend ollama --methods writer_only --num_topics 20 --model_config balanced_writer --retrieval_manager wiki --writing_mode full_article
echo "Experiment 2 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/A2_balanced_writer--writer_only--wm-full_article--revm-pending-sections--rm-wiki_03-11_1605"
echo "Completed experiment 2 with evaluation"
echo "---"

echo "All experiments completed! Total: 2"
