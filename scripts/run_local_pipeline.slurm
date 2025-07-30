#!/bin/bash
#SBATCH --job-name=local_qwen_pipeline
#SBATCH --output=logs/local_qwen_%j.out
#SBATCH --error=logs/local_qwen_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Job configuration
MODEL_SIZE=${1:-"32b"}  # Default to 32b, can pass 14b, 32b, or 72b
TOPIC_LIMIT=${2:-5}    # Default to 10 topics
EXPERIMENT_NAME="local_qwen${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"

echo "=================================================="
echo "Local Qwen Pipeline Job"
echo "Model: Qwen2.5-${MODEL_SIZE} / Qwen3-14B"
echo "Topics: ${TOPIC_LIMIT}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "=================================================="

# Setup
cd /storage/ukp/work/ibrahim1/Writer-Reviewer
source /storage/ukp/work/ibrahim1/python_env/bin/activate

# Create logs directory
mkdir -p logs

# Set environment variables to avoid permission issues
export TRANSFORMERS_CACHE=/tmp/transformers_cache_$SLURM_JOB_ID
export HF_HOME=/tmp/hf_cache_$SLURM_JOB_ID
mkdir -p $TRANSFORMERS_CACHE $HF_HOME

echo "Starting local model generation..."
echo "Time: $(date)"

# Step 1: Run local model generation
python -m src.local \
    --model_size ${MODEL_SIZE} \
    --topic_limit ${TOPIC_LIMIT} \
    --experiment_name ${EXPERIMENT_NAME} \
    --output_dir results/local \
    --log_level INFO

if [ $? -ne 0 ]; then
    echo "ERROR: Local model generation failed!"
    exit 1
fi

echo "Local generation completed at: $(date)"

# Step 2: Run evaluation
echo "Starting evaluation..."
RESULTS_DIR="results/local/${EXPERIMENT_NAME}"

python -m src.evaluation \
    --results_file "${RESULTS_DIR}/results.json" \
    --output_dir "${RESULTS_DIR}/evaluation" \
    --methods direct

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo "Evaluation completed at: $(date)"

# Step 3: Run analysis
echo "Starting analysis..."

python -m src.analysis \
    --results_dir "${RESULTS_DIR}" \
    --output_dir "${RESULTS_DIR}/analysis" \
    --experiment_name ${EXPERIMENT_NAME}

if [ $? -ne 0 ]; then
    echo "ERROR: Analysis failed!"
    exit 1
fi

echo "Analysis completed at: $(date)"

# Clean up temporary directories
rm -rf $TRANSFORMERS_CACHE $HF_HOME

echo "=================================================="
echo "Pipeline completed successfully!"
echo "Results saved in: ${RESULTS_DIR}"
echo "Total time: $(date)"
echo "=================================================="

# Print summary
echo ""
echo "SUMMARY:"
echo "- Model: Qwen2.5-${MODEL_SIZE}"
echo "- Topics processed: ${TOPIC_LIMIT}"
echo "- Results: ${RESULTS_DIR}/results.json"
echo "- Evaluation: ${RESULTS_DIR}/evaluation/"
echo "- Analysis: ${RESULTS_DIR}/analysis/"
echo ""
echo "To view results:"
echo "ls -la ${RESULTS_DIR}/"

# Configuration
MODEL_SIZE="32b"  # Options: 32b, 72b, 14b
TOPIC_LIMIT=10
EXPERIMENT_NAME="local_qwen_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"

# Create logs directory
mkdir -p logs

# Activate environment
source /storage/ukp/work/ibrahim1/python_env/bin/activate

# Change to project directory
cd /storage/ukp/work/ibrahim1/Writer-Reviewer

echo "Starting Local Qwen Pipeline"
echo "Model: Qwen ${MODEL_SIZE}"
echo "Topics: ${TOPIC_LIMIT}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "================================"

# Step 1: Run local model generation
echo "Step 1: Running local model generation..."
python -m src.local \
    --model_size ${MODEL_SIZE} \
    --topic_limit ${TOPIC_LIMIT} \
    --experiment_name ${EXPERIMENT_NAME} \
    --log_level INFO

if [ $? -ne 0 ]; then
    echo "ERROR: Local model generation failed"
    exit 1
fi

echo "Step 1 completed successfully"

# Step 2: Run evaluation
echo "Step 2: Running evaluation..."
python -m src.evaluation \
    --results_path "results/local/${EXPERIMENT_NAME}/results.json" \
    --output_dir "results/local/${EXPERIMENT_NAME}/evaluation" \
    --methods direct

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo "Step 2 completed successfully"

# Step 3: Run analysis
echo "Step 3: Running analysis..."
python -m src.analysis \
    --results_path "results/local/${EXPERIMENT_NAME}/results.json" \
    --eval_path "results/local/${EXPERIMENT_NAME}/evaluation" \
    --output_dir "results/local/${EXPERIMENT_NAME}/analysis" \
    --experiment_name ${EXPERIMENT_NAME}

if [ $? -ne 0 ]; then
    echo "ERROR: Analysis failed"
    exit 1
fi

echo "Step 3 completed successfully"

echo "================================"
echo "Pipeline completed successfully!"
echo "Results saved in: results/local/${EXPERIMENT_NAME}/"
echo "- Generation results: results/local/${EXPERIMENT_NAME}/results.json"
echo "- Evaluation results: results/local/${EXPERIMENT_NAME}/evaluation/"
echo "- Analysis results: results/local/${EXPERIMENT_NAME}/analysis/"
echo "================================"
