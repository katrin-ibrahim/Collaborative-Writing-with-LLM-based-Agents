#!/bin/bash
#SBATCH --job-name=writer_32b
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/writer_32b_%j.out
#SBATCH --error=logs/writer_32b_%j.err

# Create logs directory

cd /storage/ukp/work/ibrahim1/Collaborative-Writing-with-LLM-based-Agents/

# Activate your Python 3.11 virtual environment
source /storage/ukp/work/ibrahim1/python_env/bin/activate

mkdir -p logs

echo "Job started at $(date)"
echo "Available memory: $(free -h)"
echo "GPU info:"
nvidia-smi


echo "Starting model loading at $(date)"
# Run the experiment
python -m src.baselines --backend local -n 5 -m direct --model_size 32b
echo "Job finished at $(date)"