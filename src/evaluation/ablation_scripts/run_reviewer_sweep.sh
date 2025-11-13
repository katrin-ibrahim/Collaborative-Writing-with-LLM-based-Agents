#!/usr/bin/env bash
set -euo pipefail

# This script runs the 4-model reviewer ablation.
# It uses 'qwen2.5:32b' as the override model for all other roles.
# The reviewer_model is specified individually for each experiment.

# Total experiments: 4
echo "Starting 4-model reviewer ablation..."

# 1. Baseline Control
echo "[1/4] Running experiment: reviewer_sweep_qwen2-5_32b"
echo " -> model: qwen2.5:32b"
mkdir -p "results/ollama/reviewer_sweep_qwen2-5_32b"
python -m src.main -en "reviewer_sweep_qwen2-5_32b" \
    --model_config phase1_best \
    --reviewer_model "qwen2.5:32b" \
    --backend ollama \
    --override_model "qwen2.5:32b" \
    --methods writer_reviewer \
    --num_topics 20 >> "results/ollama/reviewer_sweep_qwen2-5_32b/main_run.log" 2>&1
echo "Generation finished; running evaluation for reviewer_sweep_qwen2-5_32b..."
python -m src.evaluation "results/ollama/reviewer_sweep_qwen2-5_32b" >> "results/ollama/reviewer_sweep_qwen2-5_32b/eval_run.log" 2>&1
echo "[1/4] Completed: reviewer_sweep_qwen2-5_32b"

# 2. Reasoning Specialist
echo "[2/4] Running experiment: reviewer_sweep_phi4"
echo " -> model: phi4:latest"
mkdir -p "results/ollama/reviewer_sweep_phi4"
python -m src.main -en "reviewer_sweep_phi4" \
    --model_config phase1_best \
    --reviewer_model "phi4:latest" \
    --backend ollama \
    --override_model "qwen2.5:32b" \
    --methods writer_reviewer \
    --num_topics 20 >> "results/ollama/reviewer_sweep_phi4/main_run.log" 2>&1
echo "Generation finished; running evaluation for reviewer_sweep_phi4..."
python -m src.evaluation "results/ollama/reviewer_sweep_phi4" >> "results/ollama/reviewer_sweep_phi4/eval_run.log" 2>&1
echo "[2/4] Completed: reviewer_sweep_phi4"

# 3. JSON/Tool-Use Specialist
echo "[3/4] Running experiment: reviewer_sweep_gemma3-tools_27b"
echo " -> model: PetrosStav/gemma3-tools:27b"
mkdir -p "results/ollama/reviewer_sweep_gemma3-tools_27b"
python -m src.main -en "reviewer_sweep_gemma3-tools_27b" \
    --model_config phase1_best \
    --reviewer_model "PetrosStav/gemma3-tools:27b" \
    --backend ollama \
    --override_model "qwen2.5:32b" \
    --methods writer_reviewer \
    --num_topics 20 >> "results/ollama/reviewer_sweep_gemma3-tools_27b/main_run.log" 2>&1
echo "Generation finished; running evaluation for reviewer_sweep_gemma3-tools_27b..."
python -m src.evaluation "results/ollama/reviewer_sweep_gemma3-tools_27b" >> "results/ollama/reviewer_sweep_gemma3-tools_27b/eval_run.log" 2>&1
echo "[3/4] Completed: reviewer_sweep_gemma3-tools_27b"

# 4. S-Tier Critic
echo "[4/4] Running experiment: reviewer_sweep_llama3-3"
echo " -> model: llama3.3:latest"
mkdir -p "results/ollama/reviewer_sweep_llama3-3"
python -m src.main -en "reviewer_sweep_llama3-3" \
    --model_config phase1_best \
    --reviewer_model "llama3.3:latest" \
    --backend ollama \
    --override_model "qwen2.5:32b" \
    --methods writer_reviewer \
    --num_topics 20 >> "results/ollama/reviewer_sweep_llama3-3/main_run.log" 2>&1
echo "Generation finished; running evaluation for reviewer_sweep_llama3-3..."
python -m src.evaluation "results/ollama/reviewer_sweep_llama3-3" >> "results/ollama/reviewer_sweep_llama3-3/eval_run.log" 2>&1
echo "[4/4] Completed: reviewer_sweep_llama3-3"

echo "All 4 experiments completed."
