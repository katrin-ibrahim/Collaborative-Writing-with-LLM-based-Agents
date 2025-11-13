#!/usr/bin/env bash
set -euo pipefail

# Generated experiment script for agentic methods: agent_exp
# Generated on: 2025-11-10 13:34:36

echo "Starting experiment 1: baseline_rag_direct"
echo "Results will be saved to: results/ollama/baseline_rag_direct"
mkdir -p "results/ollama/baseline_rag_direct"
# python -m src.main -en baseline_rag_direct --backend ollama --methods storm --num_topics 100 --retrieval_manager wiki --model_config balanced_writer  2>&1 | tee "results/ollama/baseline_rag_direct/run.log"
echo "Experiment 1 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/baseline_rag_direct" 2>&1 | tee -a "results/ollama/baseline_rag_direct/run.log"
echo "Completed experiment 1 with evaluation"
echo "---"


echo "All experiments completed! Total: 2"
