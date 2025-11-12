#!/usr/bin/env bash
set -euo pipefail

# Generated: 2025-11-10T14:44:38.597117
# Experiments: 13

echo "Starting experiment 1: baseline_32b"
echo "Description: Baseline: all tasks use 32b"
echo "Results will be saved to: results/ollama/model_ablation_baseline_32b"
mkdir -p "results/ollama/model_ablation_baseline_32b" && python -m src.main -m writer_reviewer -n 10 -b ollama -om qwen2.5:32b --experiment_name model_ablation_baseline_32b 2>&1 | tee -a "results/ollama/model_ablation_baseline_32b/run.log"
echo "Experiment 1 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_baseline_32b" 2>&1 | tee -a "results/ollama/model_ablation_baseline_32b/run.log"
echo "Completed experiment 1 with evaluation"
echo "---"

echo "Starting experiment 2: query_generation_14b"
echo "Description: Downgrade query_generation to 14b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_query_generation_14b"
mkdir -p "results/ollama/model_ablation_query_generation_14b" && python -m src.main -m writer_reviewer -n 10 -b ollama -om qwen2.5:32b --experiment_name model_ablation_query_generation_14b -qgm qwen2.5:14b 2>&1 | tee -a "results/ollama/model_ablation_query_generation_14b/run.log"
echo "Experiment 2 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_query_generation_14b" 2>&1 | tee -a "results/ollama/model_ablation_query_generation_14b/run.log"
echo "Completed experiment 2 with evaluation"
echo "---"

echo "Starting experiment 3: query_generation_7b"
echo "Description: Downgrade query_generation to 7b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_query_generation_7b"
mkdir -p "results/ollama/model_ablation_query_generation_7b" && python -m src.main -m writer_reviewer -n 10 -b ollama -om qwen2.5:32b --experiment_name model_ablation_query_generation_7b -qgm qwen2.5:7b 2>&1 | tee -a "results/ollama/model_ablation_query_generation_7b/run.log"
echo "Experiment 3 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_query_generation_7b" 2>&1 | tee -a "results/ollama/model_ablation_query_generation_7b/run.log"
echo "Completed experiment 3 with evaluation"
echo "---"

echo "Starting experiment 4: create_outline_14b"
echo "Description: Downgrade create_outline to 14b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_create_outline_14b"
mkdir -p "results/ollama/model_ablation_create_outline_14b" && python -m src.main -m writer_reviewer -n 10 -b ollama -om qwen2.5:32b --experiment_name model_ablation_create_outline_14b -oum qwen2.5:14b 2>&1 | tee -a "results/ollama/model_ablation_create_outline_14b/run.log"
echo "Experiment 4 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_create_outline_14b" 2>&1 | tee -a "results/ollama/model_ablation_create_outline_14b/run.log"
echo "Completed experiment 4 with evaluation"
echo "---"

echo "Starting experiment 5: create_outline_7b"
echo "Description: Downgrade create_outline to 7b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_create_outline_7b"
mkdir -p "results/ollama/model_ablation_create_outline_7b" && python -m src.main -m writer_reviewer -n 10 -b ollama -om qwen2.5:32b --experiment_name model_ablation_create_outline_7b -oum qwen2.5:7b 2>&1 | tee -a "results/ollama/model_ablation_create_outline_7b/run.log"
echo "Experiment 5 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_create_outline_7b" 2>&1 | tee -a "results/ollama/model_ablation_create_outline_7b/run.log"
echo "Completed experiment 5 with evaluation"
echo "---"

echo "Starting experiment 6: section_selection_14b"
echo "Description: Downgrade section_selection to 14b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_section_selection_14b"
mkdir -p "results/ollama/model_ablation_section_selection_14b" && python -m src.main -m writer_reviewer -n 10 -b ollama -om qwen2.5:32b --experiment_name model_ablation_section_selection_14b -ssm qwen2.5:14b 2>&1 | tee -a "results/ollama/model_ablation_section_selection_14b/run.log"
echo "Experiment 6 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_section_selection_14b" 2>&1 | tee -a "results/ollama/model_ablation_section_selection_14b/run.log"
echo "Completed experiment 6 with evaluation"
echo "---"

echo "Starting experiment 7: section_selection_7b"
echo "Description: Downgrade section_selection to 7b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_section_selection_7b"
mkdir -p "results/ollama/model_ablation_section_selection_7b" && python -m src.main -m writer_reviewer -n 10 -b ollama -om qwen2.5:32b --experiment_name model_ablation_section_selection_7b -ssm qwen2.5:7b 2>&1 | tee -a "results/ollama/model_ablation_section_selection_7b/run.log"
echo "Experiment 7 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_section_selection_7b" 2>&1 | tee -a "results/ollama/model_ablation_section_selection_7b/run.log"
echo "Completed experiment 7 with evaluation"
echo "---"

echo "Starting experiment 8: revision_14b"
echo "Description: Downgrade revision to 14b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_revision_14b"
mkdir -p "results/ollama/model_ablation_revision_14b" && python -m src.main -m writer_reviewer -n 10 -b ollama -om qwen2.5:32b --experiment_name model_ablation_revision_14b -rvm qwen2.5:14b 2>&1 | tee -a "results/ollama/model_ablation_revision_14b/run.log"
echo "Experiment 8 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_revision_14b" 2>&1 | tee -a "results/ollama/model_ablation_revision_14b/run.log"
echo "Completed experiment 8 with evaluation"
echo "---"

echo "Starting experiment 9: revision_7b"
echo "Description: Downgrade revision to 7b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_revision_7b"
mkdir -p "results/ollama/model_ablation_revision_7b" && python -m src.main -m writer_reviewer -n 5 -b ollama -om qwen2.5:32b --experiment_name model_ablation_revision_7b -rvm qwen2.5:7b > "results/ollama/model_ablation_revision_7b/run.log" 2>&1
echo "Experiment 9 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_revision_7b" >> "results/ollama/model_ablation_revision_7b/run.log" 2>&1
echo "Completed experiment 9 with evaluation"
echo "---"

echo "Starting experiment 10: revision_batch_14b"
echo "Description: Downgrade revision_batch to 14b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_revision_batch_14b"
mkdir -p "results/ollama/model_ablation_revision_batch_14b" && python -m src.main -m writer_reviewer -n 5 -b ollama -om qwen2.5:32b --experiment_name model_ablation_revision_batch_14b -rbm qwen2.5:14b > "results/ollama/model_ablation_revision_batch_14b/run.log" 2>&1
echo "Experiment 10 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_revision_batch_14b" >> "results/ollama/model_ablation_revision_batch_14b/run.log" 2>&1
echo "Completed experiment 10 with evaluation"
echo "---"

echo "Starting experiment 11: revision_batch_7b"
echo "Description: Downgrade revision_batch to 7b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_revision_batch_7b"
mkdir -p "results/ollama/model_ablation_revision_batch_7b" && python -m src.main -m writer_reviewer -n 5 -b ollama -om qwen2.5:32b --experiment_name model_ablation_revision_batch_7b -rbm qwen2.5:7b > "results/ollama/model_ablation_revision_batch_7b/run.log" 2>&1
echo "Experiment 11 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_revision_batch_7b" >> "results/ollama/model_ablation_revision_batch_7b/run.log" 2>&1
echo "Completed experiment 11 with evaluation"
echo "---"

echo "Starting experiment 12: self_refine_14b"
echo "Description: Downgrade self_refine to 14b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_self_refine_14b"
mkdir -p "results/ollama/model_ablation_self_refine_14b" && python -m src.main -m writer_reviewer -n 5 -b ollama -om qwen2.5:32b --experiment_name model_ablation_self_refine_14b -srm qwen2.5:14b > "results/ollama/model_ablation_self_refine_14b/run.log" 2>&1
echo "Experiment 12 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_self_refine_14b" >> "results/ollama/model_ablation_self_refine_14b/run.log" 2>&1
echo "Completed experiment 12 with evaluation"
echo "---"

echo "Starting experiment 13: self_refine_7b"
echo "Description: Downgrade self_refine to 7b (others at 32b)"
echo "Results will be saved to: results/ollama/model_ablation_self_refine_7b"
mkdir -p "results/ollama/model_ablation_self_refine_7b" && python -m src.main -m writer_reviewer -n 5 -b ollama -om qwen2.5:32b --experiment_name model_ablation_self_refine_7b -srm qwen2.5:7b > "results/ollama/model_ablation_self_refine_7b/run.log" 2>&1
echo "Experiment 13 generation completed, running evaluation..."
python -m src.evaluation "results/ollama/model_ablation_self_refine_7b" >> "results/ollama/model_ablation_self_refine_7b/run.log" 2>&1
echo "Completed experiment 13 with evaluation"
echo "---"

echo "All experiments completed! Total: 13"
