#!/usr/bin/env bash
set -euo pipefail

# This script runs the 3 specialist model ablations against the 'qwen2.5:32b' baselines.
#
# It tests:
# 1. research_model (using 'writer_v3' method)
# 2. self_refine_model (using 'writer_v3' method)
# 3. revision_model (using 'writer_reviewer' method)
#
# Baselines for comparison:
# - For Tests 1 & 2: Use 'writer_sweep_qwen2-5_32b' (which uses writer_v3)
# - For Test 3: Use 'reviewer_sweep_qwen2-5_32b' (which uses writer_reviewer)

# Total experiments: 3
echo "Starting 3-model specialist ablation..."

# 1. Test Research Model (Knowledge)
echo "[1/3] Running experiment: specialist_sweep_research_llama3"
echo " -> model: research_model=llama3.3:latest (method=writer_v3)"
mkdir -p "results/ollama/specialist_sweep_research_llama3"
python -m src.main -en "specialist_sweep_research_llama3" \
    --research_model "llama3.3:latest" \
    --backend ollama \
    --override_model "qwen2.5:32b" \
    --methods writer_v3 \
    --num_topics 20 \
    -wm section >> "results/ollama/specialist_sweep_research_llama3/main_run.log" 2>&1
echo "Generation finished; running evaluation for specialist_sweep_research_llama3..."
python -m src.evaluation "results/ollama/specialist_sweep_research_llama3" >> "results/ollama/specialist_sweep_research_llama3/eval_run.log" 2>&1
echo "[1/3] Completed: specialist_sweep_research_llama3"


# 3. Test Revision Model (Complex Reasoning)
echo "[3/3] Running experiment: specialist_sweep_revision_gemma3"
echo " -> model: revision_model=gemma3:27b (method=writer_reviewer)"
mkdir -p "results/ollama/specialist_sweep_revision_gemma3"
python -m src.main -en "specialist_sweep_revision_gemma3" \
    --revision_model "gemma3:27b" \
    --backend ollama \
    --override_model "qwen2.5:32b" \
    --methods writer_reviewer \
    --num_topics 20 >> "results/ollama/specialist_sweep_revision_gemma3/main_run.log" 2>&1
echo "Generation finished; running evaluation for specialist_sweep_revision_gemma3..."
python -m src.evaluation "results/ollama/specialist_sweep_revision_gemma3" >> "results/ollama/specialist_sweep_revision_gemma3/eval_run.log" 2>&1
echo "[3/3] Completed: specialist_sweep_revision_gemma3"

echo "All 3 specialist experiments completed."
