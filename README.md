# AI Writer Agent Framework

This project implements a flexible AI writing framework with multiple agent approaches for generating articles on various topics. It features a collaborative writing system with separate Writer and Reviewer agents that interact through a shared memory and an iterative revision protocol.

## System Architecture

Our system comprises two clearly separated agents that interact through a shared memory and an iterative revision protocol:

- **Writer**: Responsible solely for content creation. It issues retrieval queries to an external search engine, collates the top-k passages into a working context, drafts a hierarchical outline, and then expands each heading.

- **Reviewer Agent**: Comprised of three steps:

  - **Fact-Checking**: Flags unsupported or inaccurate claims
  - **Structure-Advis**: Assesses logical flow and section ordering
  - **Leader**: Merges their comments, removes duplicates, and outputs a unified review, with each comment tagged by category and severity

- **Revision Loop**: After receiving the review, the Writer revises the draft, marking each comment as addressed or contested. Convergence is the proportion of addressed comments; the loop stops when convergence exceeds 90% or after N iterations. At each turn, the Writer and Reviewer begin with a Theory of Mind prediction of the partner's priorities.

## Modular Pipeline Architecture

The system is organized into three main modules that can be run independently or together:

1. **📝 Baselines Module** (`src/baselines/`) - Generates articles using different methods
2. **🔍 Evaluation Module** (`src/evaluation/`) - Evaluates generated articles against reference content
3. **📊 Analysis Module** (`src/analysis/`) - Analyzes evaluation results and generates insights

## Agent Approaches

### Baselines

1. **STORM**: Using the official implemention of storm.
2. **Direct Prompting**: Single-prompt article generation
3. **RAG**: Retrieval-augmented generation approach

### Collaborative Approach

The collaborative approach combines a Writer agent with a Reviewer Agent to iteratively improve the draft through a revision process.

## Getting Started

### Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your HF token (optional)
export HF_TOKEN=your_huggingface_token
export PYTHONPATH=${PYTHONPATH}:./
```

### Quick Start

The easiest way to run the full pipeline is using the master script:

```bash
# Run complete pipeline (baselines → evaluation → analysis)
./scripts/run_pipeline.sh

# Run with specific methods and topics
./scripts/run_pipeline.sh -m "direct storm" -n 5

# Run only baselines (no evaluation or analysis)
./scripts/run_pipeline.sh --baselines-only -m direct -n 1
```

### Module Usage

#### 1. Running Baselines

Generate articles using different methods:

```bash
# Run all baseline methods
python -m src.baselines --num_topics 5

# Run specific method
python -m src.baselines --methods direct --num_topics 1

# Use custom Ollama host
python -m src.baselines --methods direct --num_topics 1 --ollama_host http://localhost:11434/
```

**Results Directory Format:** `results/ollama/{method(s)}_N={num_topics}_T=d{dd.mm_HH:MM}`

Example: `results/ollama/direct_N=1_T=d16.07_12:24`

#### 2. Running Evaluation

Evaluate generated articles against reference content:

```bash
# Evaluate articles in a results directory
python -m src.evaluation results/ollama/direct_N=1_T=d16.07_12:24

# Force re-evaluation even if results exist
python -m src.evaluation results/ollama/direct_N=1_T=d16.07_12:24 --force

# Evaluate with debug logging
python -m src.evaluation results/ollama/direct_N=1_T=d16.07_12:24 --log_level DEBUG
```

#### 3. Running Analysis

Analyze evaluation results and generate insights:

```bash
# Analyze results from a directory
python -m src.analysis results/ollama/M=direct_N=1_T=d16.07_12:24
```

**Analysis Output:** Creates `analysis_output/` directory with:

- `aggregated_metrics.json` - Summary statistics
- `*.png` - Visualization plots
- Detailed breakdowns by method and topic

### Pipeline Script Options

The master pipeline script (`scripts/run_pipeline.sh`) supports various options:

```bash
# Full pipeline options
./scripts/run_pipeline.sh [OPTIONS]

# Available options:
-m, --methods METHODS       # Methods to run (default: "direct storm rag")
-n, --num_topics NUM        # Number of topics (default: 5)
-H, --ollama_host HOST      # Ollama server URL
-l, --log_level LEVEL       # Log level (DEBUG, INFO, WARNING, ERROR)
--baselines-only            # Only run baselines
--evaluation-only DIR       # Only run evaluation on specified directory
--analysis-only DIR         # Only run analysis on specified directory
--force-eval                # Force re-evaluation even if results exist
```

### Examples

```bash
# Run full pipeline with direct method only
./scripts/run_pipeline.sh -m direct -n 1

# Run evaluation only on existing results
./scripts/run_pipeline.sh --evaluation-only results/ollama/M=direct_N=1_T=d16.07_12:24

# Run analysis only on existing results
./scripts/run_pipeline.sh --analysis-only results/ollama/M=direct_N=1_T=d16.07_12:24

# Run baselines with multiple methods
./scripts/run_pipeline.sh --baselines-only -m "direct storm rag" -n 5

## Evaluation

The system evaluates content quality using several metrics:

- **Heading Soft Recall (HSR)**: Sentence-BERT embeddings measure topic coverage by averaging the best cosine similarity between gold and generated headings.
- **Heading Entity Recall (HER)**: FLAIR NER checks what fraction of named entities in reference headings re-appear in model heading.
- **ROUGE-1/2/L**: n-gram overlap between generated and gold articles, computed per section after each revision to capture quality improvements over iterations.
- **Article Entity Recall (AER)**: FLAIR-based recall of all named entities in the reference article.

## Results Directory Structure

The system creates organized results directories with the following structure:

```

results/ollama/M={method(s)}\_N={topics}\_T=d{dd.mm_HH:MM}/
├── articles/ # Generated articles (markdown files)
│ ├── direct_Topic_Name.md
│ ├── storm_Topic_Name.md
│ └── rag_Topic_Name.md
├── results.json # Experiment results and configuration
├── analysis_output/ # Analysis results (created by analysis module)
│ ├── aggregated_metrics.json
│ ├── distributions.png
│ ├── effect_sizes.png
│ ├── metric_comparison.png
│ └── success_analysis.png
└── checkpoint.json # Progress tracking (for resuming)

````

### Directory Naming Format

The new results directory naming format is designed to be self-documenting:

**Format:** `M={method(s)}_N={num_topics}_T=d{dd.mm_HH:MM}`

**Examples:**
- `M=direct_N=1_T=d16.07_12:24` → Direct method, 1 topic, July 16th at 12:24
- `M=direct_storm_N=5_T=d16.07_14:30` → Direct+Storm methods, 5 topics, July 16th at 14:30
- `M=direct_rag_storm_N=10_T=d17.07_09:15` → All methods, 10 topics, July 17th at 09:15

**Benefits:**
- **M=** prefix clearly indicates the method(s) used
- **N=** prefix shows the number of topics/samples
- **T=d** prefix shows the date and time (with `d` for day)
- Multiple methods are sorted alphabetically for consistency
- Filesystem-safe characters (no spaces or special characters)

## Migration from Old Scripts

**⚠️ Deprecated Files:**
- `regenerate_evaluations.py` - Replaced by `python -m src.evaluation`
- `run_analysis.py` - Replaced by `python -m src.analysis`
- `main.py` - Replaced by `python -m src.baselines`

**Migration Guide:**
```bash
# Old way
python main.py --method direct --num_topics 5
python regenerate_evaluations.py
python run_analysis.py

# New way (individual modules)
python -m src.baselines --methods direct --num_topics 5
python -m src.evaluation results/ollama/M=direct_N=5_T=d16.07_12:24
python -m src.analysis results/ollama/M=direct_N=5_T=d16.07_12:24

# New way (pipeline script)
./scripts/run_pipeline.sh -m direct -n 5
````

## Development

- [x] **NEW**: Modular architecture with separate baselines, evaluation, and analysis modules
- [x] **NEW**: Master pipeline script for orchestrating the full workflow
- [x] **NEW**: Improved results directory naming with clear method/topic/time format

### Testing

```bash
# Test individual modules
python -m src.baselines --help
python -m src.evaluation --help
python -m src.analysis --help

# Test pipeline script
./scripts/run_pipeline.sh --help
```

### Advanced Usage

#### Resuming Experiments

You can resume interrupted experiments using the `--resume_dir` flag:

```bash
# Resume from specific directory
python -m src.baselines --resume_dir results/ollama/M=direct_N=5_T=d16.07_12:24
```

#### Custom Configuration

```bash
# Use custom model configuration
python -m src.baselines --model_config config/custom_models.yaml

# Enable debug mode (saves intermediate files)
python -m src.baselines --debug --log_level DEBUG
```

## SLURM Setup (HPC Clusters)

For running on HPC clusters with SLURM:

#### First time setup:

```bash
cd /storage/ukp/work/ibrahim1
wget https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz
tar -xzf Python-3.11.8.tgz
cd Python-3.11.8
./configure --prefix=/storage/ukp/work/ibrahim1/python3.11 --enable-optimizations
make -j$(nproc)
make install
# Add python to path
export PATH="/storage/ukp/work/ibrahim1/python3.11/bin:$PATH"
# Verify it works
python3.11 --version
pip3.11 --version
# Create virtual environment with your new Python 3.11
python3.11 -m venv /storage/ukp/work/ibrahim1/python_env
# Activate the environment
source /storage/ukp/work/ibrahim1/python_env/bin/activate
```

#### Running experiments:

```bash
cd /storage/ukp/work/ibrahim1
source /storage/ukp/work/ibrahim1/python_env/bin/activate
# To submit a job
$ sbatch -q yolo -p yolo run.sh

# Interactive session
srun --mem=16G --cpus-per-task=4 --gres=gpu:1 --time=01:00:00 --pty bash
source /storage/ukp/work/ibrahim1/python_env/bin/activate
./scripts/run_pipeline.sh -m direct -n 5 --baselines-only
```
