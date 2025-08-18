#!/bin/bash
# Generated experiment script: exp
# Generated on: 2025-08-16 20:13:09



echo "Starting experiment 5: exp_005_n5__c-ollama_localhost__rm-bm25_wiki"
echo "Results will be saved to: results/ollama/exp_005_n5__c-ollama_localhost__rm-bm25_wiki"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager bm25_wiki --experiment_name exp_005_n5__c-ollama_localhost__rm-bm25_wiki
echo "Experiment 5 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_005_n5__c-ollama_localhost__rm-bm25_wiki"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_005_n5__c-ollama_localhost__rm-bm25_wiki"
echo "Completed experiment 5 with evaluation and analysis"
echo "---"

echo "Starting experiment 6: exp_006_n5__c-ollama_localhost__rm-bm25_wiki__wd"
echo "Results will be saved to: results/ollama/exp_006_n5__c-ollama_localhost__rm-bm25_wiki__wd"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager bm25_wiki --use_wikidata_enhancement --experiment_name exp_006_n5__c-ollama_localhost__rm-bm25_wiki__wd
echo "Experiment 6 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_006_n5__c-ollama_localhost__rm-bm25_wiki__wd"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_006_n5__c-ollama_localhost__rm-bm25_wiki__wd"
echo "Completed experiment 6 with evaluation and analysis"
echo "---"

echo "Starting experiment 7: exp_007_n5__c-ollama_localhost__rm-bm25_wiki__sf"
echo "Results will be saved to: results/ollama/exp_007_n5__c-ollama_localhost__rm-bm25_wiki__sf"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager bm25_wiki --semantic_filtering --experiment_name exp_007_n5__c-ollama_localhost__rm-bm25_wiki__sf
echo "Experiment 7 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_007_n5__c-ollama_localhost__rm-bm25_wiki__sf"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_007_n5__c-ollama_localhost__rm-bm25_wiki__sf"
echo "Completed experiment 7 with evaluation and analysis"
echo "---"

echo "Starting experiment 8: exp_008_n5__c-ollama_localhost__rm-bm25_wiki__sf__wd"
echo "Results will be saved to: results/ollama/exp_008_n5__c-ollama_localhost__rm-bm25_wiki__sf__wd"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager bm25_wiki --semantic_filtering --use_wikidata_enhancement --experiment_name exp_008_n5__c-ollama_localhost__rm-bm25_wiki__sf__wd
echo "Experiment 8 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_008_n5__c-ollama_localhost__rm-bm25_wiki__sf__wd"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_008_n5__c-ollama_localhost__rm-bm25_wiki__sf__wd"
echo "Completed experiment 8 with evaluation and analysis"
echo "---"

echo "Starting experiment 9: exp_009_n5__c-ollama_localhost__rm-faiss_wiki"
echo "Results will be saved to: results/ollama/exp_009_n5__c-ollama_localhost__rm-faiss_wiki"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager faiss_wiki --experiment_name exp_009_n5__c-ollama_localhost__rm-faiss_wiki
echo "Experiment 9 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_009_n5__c-ollama_localhost__rm-faiss_wiki"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_009_n5__c-ollama_localhost__rm-faiss_wiki"
echo "Completed experiment 9 with evaluation and analysis"
echo "---"

echo "Starting experiment 10: exp_010_n5__c-ollama_localhost__rm-faiss_wiki__wd"
echo "Results will be saved to: results/ollama/exp_010_n5__c-ollama_localhost__rm-faiss_wiki__wd"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager faiss_wiki --use_wikidata_enhancement --experiment_name exp_010_n5__c-ollama_localhost__rm-faiss_wiki__wd
echo "Experiment 10 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_010_n5__c-ollama_localhost__rm-faiss_wiki__wd"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_010_n5__c-ollama_localhost__rm-faiss_wiki__wd"
echo "Completed experiment 10 with evaluation and analysis"
echo "---"

echo "Starting experiment 11: exp_011_n5__c-ollama_localhost__rm-faiss_wiki__sf"
echo "Results will be saved to: results/ollama/exp_011_n5__c-ollama_localhost__rm-faiss_wiki__sf"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager faiss_wiki --semantic_filtering --experiment_name exp_011_n5__c-ollama_localhost__rm-faiss_wiki__sf
echo "Experiment 11 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_011_n5__c-ollama_localhost__rm-faiss_wiki__sf"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_011_n5__c-ollama_localhost__rm-faiss_wiki__sf"
echo "Completed experiment 11 with evaluation and analysis"
echo "---"

echo "Starting experiment 12: exp_012_n5__c-ollama_localhost__rm-faiss_wiki__sf__wd"
echo "Results will be saved to: results/ollama/exp_012_n5__c-ollama_localhost__rm-faiss_wiki__sf__wd"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_localhost --retrieval_manager faiss_wiki --semantic_filtering --use_wikidata_enhancement --experiment_name exp_012_n5__c-ollama_localhost__rm-faiss_wiki__sf__wd
echo "Experiment 12 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_012_n5__c-ollama_localhost__rm-faiss_wiki__sf__wd"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_012_n5__c-ollama_localhost__rm-faiss_wiki__sf__wd"
echo "Completed experiment 12 with evaluation and analysis"
echo "---"


echo "Starting experiment 17: exp_017_n5__c-ollama_ukp__rm-bm25_wiki"
echo "Results will be saved to: results/ollama/exp_017_n5__c-ollama_ukp__rm-bm25_wiki"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_ukp --retrieval_manager bm25_wiki --experiment_name exp_017_n5__c-ollama_ukp__rm-bm25_wiki
echo "Experiment 17 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_017_n5__c-ollama_ukp__rm-bm25_wiki"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_017_n5__c-ollama_ukp__rm-bm25_wiki"
echo "Completed experiment 17 with evaluation and analysis"
echo "---"

echo "Starting experiment 18: exp_018_n5__c-ollama_ukp__rm-bm25_wiki__wd"
echo "Results will be saved to: results/ollama/exp_018_n5__c-ollama_ukp__rm-bm25_wiki__wd"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_ukp --retrieval_manager bm25_wiki --use_wikidata_enhancement --experiment_name exp_018_n5__c-ollama_ukp__rm-bm25_wiki__wd
echo "Experiment 18 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_018_n5__c-ollama_ukp__rm-bm25_wiki__wd"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_018_n5__c-ollama_ukp__rm-bm25_wiki__wd"
echo "Completed experiment 18 with evaluation and analysis"
echo "---"

echo "Starting experiment 19: exp_019_n5__c-ollama_ukp__rm-bm25_wiki__sf"
echo "Results will be saved to: results/ollama/exp_019_n5__c-ollama_ukp__rm-bm25_wiki__sf"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_ukp --retrieval_manager bm25_wiki --semantic_filtering --experiment_name exp_019_n5__c-ollama_ukp__rm-bm25_wiki__sf
echo "Experiment 19 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_019_n5__c-ollama_ukp__rm-bm25_wiki__sf"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_019_n5__c-ollama_ukp__rm-bm25_wiki__sf"
echo "Completed experiment 19 with evaluation and analysis"
echo "---"

echo "Starting experiment 20: exp_020_n5__c-ollama_ukp__rm-bm25_wiki__sf__wd"
echo "Results will be saved to: results/ollama/exp_020_n5__c-ollama_ukp__rm-bm25_wiki__sf__wd"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_ukp --retrieval_manager bm25_wiki --semantic_filtering --use_wikidata_enhancement --experiment_name exp_020_n5__c-ollama_ukp__rm-bm25_wiki__sf__wd
echo "Experiment 20 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_020_n5__c-ollama_ukp__rm-bm25_wiki__sf__wd"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_020_n5__c-ollama_ukp__rm-bm25_wiki__sf__wd"
echo "Completed experiment 20 with evaluation and analysis"
echo "---"

echo "Starting experiment 21: exp_021_n5__c-ollama_ukp__rm-faiss_wiki"
echo "Results will be saved to: results/ollama/exp_021_n5__c-ollama_ukp__rm-faiss_wiki"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_ukp --retrieval_manager faiss_wiki --experiment_name exp_021_n5__c-ollama_ukp__rm-faiss_wiki
echo "Experiment 21 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_021_n5__c-ollama_ukp__rm-faiss_wiki"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_021_n5__c-ollama_ukp__rm-faiss_wiki"
echo "Completed experiment 21 with evaluation and analysis"
echo "---"

echo "Starting experiment 22: exp_022_n5__c-ollama_ukp__rm-faiss_wiki__wd"
echo "Results will be saved to: results/ollama/exp_022_n5__c-ollama_ukp__rm-faiss_wiki__wd"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_ukp --retrieval_manager faiss_wiki --use_wikidata_enhancement --experiment_name exp_022_n5__c-ollama_ukp__rm-faiss_wiki__wd
echo "Experiment 22 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_022_n5__c-ollama_ukp__rm-faiss_wiki__wd"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_022_n5__c-ollama_ukp__rm-faiss_wiki__wd"
echo "Completed experiment 22 with evaluation and analysis"
echo "---"

echo "Starting experiment 23: exp_023_n5__c-ollama_ukp__rm-faiss_wiki__sf"
echo "Results will be saved to: results/ollama/exp_023_n5__c-ollama_ukp__rm-faiss_wiki__sf"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_ukp --retrieval_manager faiss_wiki --semantic_filtering --experiment_name exp_023_n5__c-ollama_ukp__rm-faiss_wiki__sf
echo "Experiment 23 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_023_n5__c-ollama_ukp__rm-faiss_wiki__sf"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_023_n5__c-ollama_ukp__rm-faiss_wiki__sf"
echo "Completed experiment 23 with evaluation and analysis"
echo "---"

echo "Starting experiment 24: exp_024_n5__c-ollama_ukp__rm-faiss_wiki__sf__wd"
echo "Results will be saved to: results/ollama/exp_024_n5__c-ollama_ukp__rm-faiss_wiki__sf__wd"
python -m src.baselines --backend ollama --methods storm rag --num_topics 5 --model_config ollama_ukp --retrieval_manager faiss_wiki --semantic_filtering --use_wikidata_enhancement --experiment_name exp_024_n5__c-ollama_ukp__rm-faiss_wiki__sf__wd
echo "Experiment 24 baseline completed, running evaluation..."
python -m src.evaluation "results/ollama/exp_024_n5__c-ollama_ukp__rm-faiss_wiki__sf__wd"
echo "Evaluation completed, running analysis..."
python -m src.analysis "results/ollama/exp_024_n5__c-ollama_ukp__rm-faiss_wiki__sf__wd"
echo "Completed experiment 24 with evaluation and analysis"
echo "---"

echo "All experiments completed! Total: 24"
