#!/bin/bash
# Generated experiment script: exp
# Generated on: 2025-08-20 19:38:10
#

echo "Starting experiment 1: exp_001_n20__c-ollama_localhost__rm-wiki__sf_20-08_19:38"
echo "Results will be saved to: results/ollama/exp_001_n20__c-ollama_localhost__rm-wiki__sf_20-08_19:38"
python -m src.baselines --backend ollama --methods storm rag --num_topics 20 --model_config ollama_localhost --retrieval_manager wiki --semantic_filtering true --experiment_name exp_001_n20__c-ollama_localhost__rm-wiki__sf_20-08_19:38
echo "Experiment 1 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_001_n20__c-ollama_localhost__rm-wiki__sf_20-08_19:38"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_001_n20__c-ollama_localhost__rm-wiki__sf_20-08_19:38"
echo "Completed experiment 1 with evaluation and analysis"
echo "---"

echo "Starting experiment 2: exp_002_n20__c-ollama_localhost__rm-supabase_faiss__sf_20-08_19:38"
echo "Results will be saved to: results/ollama/exp_002_n20__c-ollama_localhost__rm-supabase_faiss__sf_20-08_19:38"
python -m src.baselines --backend ollama --methods storm rag --num_topics 20 --model_config ollama_localhost --retrieval_manager supabase_faiss --semantic_filtering true --experiment_name exp_002_n20__c-ollama_localhost__rm-supabase_faiss__sf_20-08_19:38
echo "Experiment 2 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_002_n20__c-ollama_localhost__rm-supabase_faiss__sf_20-08_19:38"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_002_n20__c-ollama_localhost__rm-supabase_faiss__sf_20-08_19:38"
echo "Completed experiment 2 with evaluation and analysis"
echo "---"

echo "All experiments completed! Total: 2"
