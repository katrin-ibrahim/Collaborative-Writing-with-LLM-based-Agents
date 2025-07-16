#!/bin/bash

# Master pipeline script for running the full experimental pipeline
# Usage: ./run_pipeline.sh [OPTIONS]

set -e  # Exit on any error

# Default values
METHODS="direct storm rag"
NUM_TOPICS=5
OLLAMA_HOST="http://10.167.31.201:11434/"
LOG_LEVEL="INFO"
RUN_BASELINES=true
RUN_EVALUATION=true
RUN_ANALYSIS=true
FORCE_EVAL=false

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Master pipeline script for running baselines, evaluation, and analysis.

OPTIONS:
    -m, --methods METHODS       Methods to run (default: "direct storm rag")
    -n, --num_topics NUM        Number of topics (default: 5)
    -H, --ollama_host HOST      Ollama server URL (default: "http://10.167.31.201:11434/")
    -l, --log_level LEVEL       Log level (default: INFO)
    --baselines-only            Only run baselines (skip evaluation and analysis)
    --evaluation-only DIR       Only run evaluation on specified results directory
    --analysis-only DIR         Only run analysis on specified results directory
    --force-eval                Force re-evaluation even if results exist
    -h, --help                  Show this help message

Examples:
    # Run full pipeline with default settings
    $0

    # Run only direct method on 10 topics
    $0 -m direct -n 10

    # Run only evaluation on existing results
    $0 --evaluation-only results/ollama/M=direct_N=1_T=d16.07_12:24

    # Run full pipeline with forced re-evaluation
    $0 --force-eval

    # Run baselines only (no evaluation or analysis)
    $0 --baselines-only
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--methods)
            METHODS="$2"
            shift 2
            ;;
        -n|--num_topics)
            NUM_TOPICS="$2"
            shift 2
            ;;
        -H|--ollama_host)
            OLLAMA_HOST="$2"
            shift 2
            ;;
        -l|--log_level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --baselines-only)
            RUN_EVALUATION=false
            RUN_ANALYSIS=false
            shift
            ;;
        --evaluation-only)
            RUN_BASELINES=false
            RUN_ANALYSIS=false
            RESULTS_DIR="$2"
            shift 2
            ;;
        --analysis-only)
            RUN_BASELINES=false
            RUN_EVALUATION=false
            RESULTS_DIR="$2"
            shift 2
            ;;
        --force-eval)
            FORCE_EVAL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "src/baselines/__main__.py" ]]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# Step 1: Run baselines
if [[ "$RUN_BASELINES" == "true" ]]; then
    log_info "🚀 Step 1: Running baselines..."
    log_info "Methods: $METHODS"
    log_info "Topics: $NUM_TOPICS"
    log_info "Ollama host: $OLLAMA_HOST"

    # Convert methods string to array format for Python
    METHODS_ARRAY=""
    for method in $METHODS; do
        METHODS_ARRAY="$METHODS_ARRAY --methods $method"
    done

    # Run baselines
    if python -m src.baselines \
        $METHODS_ARRAY \
        --num_topics $NUM_TOPICS \
        --ollama_host "$OLLAMA_HOST" \
        --log_level $LOG_LEVEL; then

        log_success "✅ Baselines completed successfully"

        # Get the most recent results directory
        RESULTS_DIR=$(find results/ollama -name "M=*" -type d | sort | tail -1)
        log_info "📂 Results directory: $RESULTS_DIR"
    else
        log_error "❌ Baselines failed"
        exit 1
    fi
fi

# Step 2: Run evaluation
if [[ "$RUN_EVALUATION" == "true" ]]; then
    log_info "🔍 Step 2: Running evaluation..."

    if [[ -z "$RESULTS_DIR" ]]; then
        log_error "No results directory specified for evaluation"
        exit 1
    fi

    # Prepare evaluation arguments
    EVAL_ARGS="--log_level $LOG_LEVEL"
    if [[ "$FORCE_EVAL" == "true" ]]; then
        EVAL_ARGS="$EVAL_ARGS --force"
    fi

    # Run evaluation
    if python -m src.evaluation "$RESULTS_DIR" $EVAL_ARGS; then
        log_success "✅ Evaluation completed successfully"
    else
        log_error "❌ Evaluation failed"
        exit 1
    fi
fi

# Step 3: Run analysis
if [[ "$RUN_ANALYSIS" == "true" ]]; then
    log_info "📊 Step 3: Running analysis..."

    if [[ -z "$RESULTS_DIR" ]]; then
        log_error "No results directory specified for analysis"
        exit 1
    fi

    # Run analysis
    if python -m src.analysis "$RESULTS_DIR"; then
        log_success "✅ Analysis completed successfully"
    else
        log_error "❌ Analysis failed"
        exit 1
    fi
fi

# Final summary
log_success "🎉 Pipeline completed successfully!"
if [[ -n "$RESULTS_DIR" ]]; then
    log_info "📂 Results location: $RESULTS_DIR"
    log_info "📊 Analysis output: $RESULTS_DIR/analysis_output"
fi
