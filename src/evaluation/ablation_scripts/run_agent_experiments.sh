#!/usr/bin/env bash
set -euo pipefail

# Generated experiment script for agentic methods: agent_exp
# Generated on: 2025-11-10 13:34:36

echo "Starting experiment 1: agent_exp_001_balanced_writer_gptoss_reviewer--writer_only--wm-section--revm-pending-sections--rm-wiki_10-11_1334"
echo "Results will be saved to: results/ollama/agent_exp_001_balanced_writer_gptoss_reviewer--writer_only--wm-section--revm-pending-sections--rm-wiki_10-11_1334"
mkdir -p "results/ollama/agent_exp_001_balanced_writer_gptoss_reviewer--writer_only--wm-section--revm-pending-sections--rm-wiki_10-11_1334"
python -m src.main -en agent_exp_001_balanced_writer_gptoss_reviewer--writer_only--wm-section--revm-pending-sections--rm-wiki_10-11_1334 --backend ollama --methods writer_only --num_topics 20 --model_config balanced_writer_gptoss_reviewer --retrieval_manager wiki --writing_mode section --no_two_phase_research 2>&1 | tee "results/ollama/agent_exp_001_balanced_writer_gptoss_reviewer--writer_only--wm-section--revm-pending-sections--rm-wiki_10-11_1334/run.log"
echo "Experiment 1 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/agent_exp_001_balanced_writer_gptoss_reviewer--writer_only--wm-section--revm-pending-sections--rm-wiki_10-11_1334" 2>&1 | tee -a "results/ollama/agent_exp_001_balanced_writer_gptoss_reviewer--writer_only--wm-section--revm-pending-sections--rm-wiki_10-11_1334/run.log"
echo "Completed experiment 1 with evaluation"
echo "---"

echo "Starting experiment 2: agent_exp_002_balanced_writer_gptoss_reviewer--writer_only--wm-full--revm-pending-sections--rm-wiki_10-11_1334"
echo "Results will be saved to: results/ollama/agent_exp_002_balanced_writer_gptoss_reviewer--writer_only--wm-full--revm-pending-sections--rm-wiki_10-11_1334"
mkdir -p "results/ollama/agent_exp_002_balanced_writer_gptoss_reviewer--writer_only--wm-full--revm-pending-sections--rm-wiki_10-11_1334"
python -m src.main -en agent_exp_002_balanced_writer_gptoss_reviewer--writer_only--wm-full--revm-pending-sections--rm-wiki_10-11_1334 --backend ollama --methods writer_only --num_topics 20 --model_config balanced_writer_gptoss_reviewer --retrieval_manager wiki --writing_mode full --no_two_phase_research 2>&1 | tee "results/ollama/agent_exp_002_balanced_writer_gptoss_reviewer--writer_only--wm-full--revm-pending-sections--rm-wiki_10-11_1334/run.log"
echo "Experiment 2 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/agent_exp_002_balanced_writer_gptoss_reviewer--writer_only--wm-full--revm-pending-sections--rm-wiki_10-11_1334" 2>&1 | tee -a "results/ollama/agent_exp_002_balanced_writer_gptoss_reviewer--writer_only--wm-full--revm-pending-sections--rm-wiki_10-11_1334/run.log"
echo "Completed experiment 2 with evaluation"
echo "---"

echo "All experiments completed! Total: 2"
