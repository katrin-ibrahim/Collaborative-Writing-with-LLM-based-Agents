#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --qos=gpu-small
#SBATCH --gres=gpu:1

cd /storage/ukp/work/ibrahim1/Collaborative-Writing-with-LLM-based-Agents/

# Activate your Python 3.11 virtual environment
source /storage/ukp/work/ibrahim1/python_env/bin/activate

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

python src/baselines_runner.py \
    --num_topics 2 \
    --methods storm \
    --log_level INFO \
    --skip_evaluation

echo "Job completed at $(date)"
