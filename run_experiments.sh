#!/bin/bash
# Generated experiment script: exp
# Generated on: 2025-08-20 10:09:37
#

echo "Starting experiment 1: exp_001_n5__c-ollama_localhost__rm-wiki__sf_20-08_13:35"
echo "Results will be saved to: results/ollama/exp_001_n5__c-ollama_localhost__rm-wiki__sf_20-08_13:35"
python -m src.baselines --backend ollama --methods rag storm --num_topics 5 --model_config ollama_localhost --retrieval_manager wiki --semantic_filtering true --experiment_name exp_001_n5__c-ollama_localhost__rm-wiki__sf_20-08_13:35 > log.txt
echo "Experiment 1 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_001_n5__c-ollama_localhost__rm-wiki__sf_20-08_13:35"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_001_n5__c-ollama_localhost__rm-wiki__sf_20-08_13:35"
echo "Completed experiment 1 with evaluation and analysis"
echo "---"

# echo "Starting experiment 2: exp_002_n5__c-ollama_localhost__rm-supabase_faiss__sf_20-08_10:09"
# echo "Results will be saved to: results/ollama/exp_002_n5__c-ollama_localhost__rm-supabase_faiss__sf_20-08_10:09"
# python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager supabase_faiss --semantic_filtering true --experiment_name exp_002_n5__c-ollama_localhost__rm-supabase_faiss__sf_20-08_10:09
# echo "Experiment 2 baseline completed, running evaluation..."
# python -m src.evaluation "results/ollama/exp_002_n5__c-ollama_localhost__rm-supabase_faiss__sf_20-08_10:09"
# echo "Evaluation completed, running analysis..."
# python -m src.analysis "results/ollama/exp_002_n5__c-ollama_localhost__rm-supabase_faiss__sf_20-08_10:09"
# echo "Completed experiment 2 with evaluation and analysis"
# echo "---"

# echo "All experiments completed! Total: 2"
