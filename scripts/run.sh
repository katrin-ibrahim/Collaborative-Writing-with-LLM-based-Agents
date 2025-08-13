#!/bin/bash


#SBATCH --job-name=baseline_exp_5_
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/baseline_exp_5_%j.out
#SBATCH --error=logs/baseline_exp_5_%j.err


# Variables
MODEL_SIZE="14b"
N_SAMPLES=1

cd /storage/ukp/work/ibrahim1/Writer-Reviewer

source /storage/ukp/work/ibrahim1/python_env/bin/activate

mkdir -p logs

echo "Job started at $(date)"
echo "Available memory: $(free -h)"
echo "GPU info:"
nvidia-smi

echo "Starting model loading at $(date)"
echo "Experiment parameters: MODEL_SIZE=${MODEL_SIZE}, N_SAMPLES=${N_SAMPLES}"
python -m src.baselines --backend local -n ${N_SAMPLES} -m rag 
echo "Job finished at $(date)"