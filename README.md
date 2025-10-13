# AI Writer Agent Framework

A flexible research framework for AI-powered collaborative writing with multiple agent approaches and comprehensive evaluation. Features both baseline methods and sophisticated collaborative Writer-Reviewer agents that interact through shared memory with iterative revision protocols.

## 🏗️ System Architecture

### Core Methods

#### **Baseline Methods**

- **Direct Prompting**: Single-prompt article generation
- **RAG (Retrieval-Augmented Generation)**: Retrieval-enhanced content generation
- **STORM**: Multi-perspective knowledge synthesis (using official implementation)

#### **Collaborative Methods**

- **Writer-Only**: Advanced multi-step writing with outline generation and section-by-section expansion
- **Writer-Reviewer**: Collaborative writing with iterative feedback and revision cycles

### **Collaborative Agent Architecture**

Our collaborative system features two specialized agents:

- **Writer Agent**:

  - Issues targeted retrieval queries to external search engines
  - Generates hierarchical outlines with strategic section planning
  - Expands each section with retrieved context
  - Revises content based on reviewer feedback

- **Reviewer Agent**:

  - **Fact-Checking**: Identifies unsupported or inaccurate claims
  - **Structure Analysis**: Evaluates logical flow and section organization
  - **Holistic Review**: Provides unified feedback with severity ratings
  - **Theory of Mind**: Predicts partner priorities for enhanced collaboration

- **Shared Memory & Revision Protocol**:
  - Persistent storage of drafts, feedback, and revision history
  - Convergence tracking (stops at 90% addressed feedback or max iterations)
  - Comprehensive interaction logging for analysis

## 🚀 Quick Start

### Setup

```bash
# Clone and setup environment
git clone https://github.com/katrin-ibrahim/Collaborative-Writing-with-LLM-based-Agents.git
cd Collaborative-Writing-with-LLM-based-Agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export HF_TOKEN=your_huggingface_token
export PYTHONPATH=${PYTHONPATH}:./
```

### Basic Usage

```bash
# Test with small model - Direct method
python -m src.main -m direct -n 1 -om gemma3:1b -b ollama -rm supabase_faiss

# Test collaborative writing
python -m src.main -m writer_reviewer -n 1 -om gemma3:1b -b ollama -rm supabase_faiss

# Run multiple methods
python -m src.main -m direct rag storm -n 5 -b ollama

# Use SLURM backend (HPC clusters)
python -m src.main -m direct -n 1 -b slurm -om distilgpt2
```

## 📋 Command Line Interface

### Core Arguments

```bash
# Backend selection
--backend, -b          # ollama (API) or slurm (direct model execution)
--num_topics, -n       # Number of topics to evaluate (default: 5)
--methods, -m          # Methods to run: direct, rag, storm, writer_only, writer_reviewer

# Model configuration
--model_config, -c     # Configuration preset: ollama_localhost, ollama_ukp, slurm, slurm_thinking
--override_model, -om  # Override all models (e.g., qwen2.5:7b, gemma3:1b, distilgpt2)

# Retrieval configuration
--retrieval_manager, -rm    # wiki or supabase_faiss
--semantic_filtering, -sf   # true/false (default: true)

# Output and control
--output_dir, -o           # Custom output directory
--experiment_name, -en     # Custom experiment name
--auto_name, -an          # Auto-generate directory names
--resume_dir, -r          # Resume from checkpoint
--debug, -d               # Enable debug mode
--log_level, -l           # DEBUG, INFO, WARNING, ERROR
```

### Configuration Presets

#### **Ollama Configurations**

- `ollama_localhost`: Resource-optimized for local development
- `ollama_ukp`: Performance-optimized for UKP server with larger models

#### **SLURM Configurations**

- `slurm`: Standard local model execution
- `slurm_thinking`: Reasoning-optimized models for complex tasks

### Task-Specific Model Assignments

Each configuration uses optimized models for different cognitive tasks:

**Ollama Localhost (Resource-Optimized)**:

- Query Generation & Selection: `llama3.2:3b` (fast, efficient)
- Section Writing: `qwen2.5:7b` (quality content generation)
- Section Revision & Review: `llama3.1:8b` (high-quality refinement)

**Ollama UKP (Performance-Optimized)**:

- Query Generation & Selection: `qwen2.5:7b` (balanced performance)
- Section Writing: `qwen2.5:32b` (maximum quality)
- Section Revision & Review: `qwen2.5:14b` (strong reasoning)

## 📖 Usage Examples

### Development & Testing

```bash
# Quick test with tiny model
python -m src.main -m direct -n 1 -om gemma3:1b -b ollama -rm supabase_faiss

# Test all methods with small models
python -m src.main -m direct rag writer_only -n 2 -om llama3.2:3b -b ollama

# Compare collaborative vs baseline
python -m src.main -m direct writer_reviewer -n 3 -b ollama
```

### Production Experiments

```bash
# High-quality collaborative writing with UKP server
python -m src.main -m writer_reviewer -n 10 -c ollama_ukp -rm supabase_faiss

# Large-scale baseline comparison
python -m src.main -m direct rag storm -n 50 -c ollama_ukp --auto_name

# SLURM cluster execution
python -m src.main -m writer_reviewer -n 20 -b slurm -c slurm_thinking
```

### Advanced Configuration

```bash
# Custom experiment with specific settings
python -m src.main \
  --methods writer_reviewer \
  --num_topics 5 \
  --backend ollama \
  --retrieval_manager supabase_faiss \
  --semantic_filtering true \
  --experiment_name "collaborative_writing_test" \
  --debug \
  --log_level DEBUG

# Resume interrupted experiment
python -m src.main --resume_dir results/ollama/writer_reviewer_N=10_T=14.10_15:30
```

### Evaluating Performance

- To evaluate the performance simply run the evaluation module with the experiment output directory as an argument

```
python -m src.evaluation results/ollama/writer_reviewer_N=10_T=14.10_15:30
```

## 🔍 Evaluation Metrics

The system evaluates content quality using several research-grade metrics:

- **Heading Soft Recall (HSR)**: Sentence-BERT embeddings measure topic coverage
- **Heading Entity Recall (HER)**: Named entity coverage in section headings
- **ROUGE-1/2/L**: n-gram overlap between generated and reference articles
- **Article Entity Recall (AER)**: Overall named entity recall across full articles
- **Collaborative Metrics**: Convergence tracking, revision effectiveness, feedback addressing rates

## 📁 Results Structure

Results are organized in self-documenting directories:

```
results/{backend}/{method}_N={topics}_T={timestamp}/
├── articles/                    # Generated articles (.md files)
│   ├── direct_Topic_Name.md
│   ├── writer_reviewer_Topic_Name.md
│   └── storm_Topic_Name.md
├── results.json                 # Experiment configuration & metadata
├── collaborative_memory/        # Writer-Reviewer interaction logs
│   ├── memory_Topic_Name.json
│   └── feedback_history.json
├── evaluation/                  # Evaluation results
│   ├── metrics.json
│   └── detailed_scores.json
└── debug/                      # Debug artifacts (if --debug enabled)
    ├── queries/
    ├── retrieved_passages/
    └── intermediate_drafts/
```

### Directory Naming Convention

**Format**: `{method}_N={num_topics}_T={dd.mm_HH:MM}`

**Examples**:

- `direct_N=5_T=14.10_15:30` → Direct method, 5 topics, Oct 14th at 15:30
- `writer_reviewer_N=10_T=14.10_09:15` → Collaborative method, 10 topics
- `multi_method_N=20_T=14.10_12:45` → Multiple methods in single run

## 🖥️ SLURM Setup (HPC Clusters)

For running on HPC clusters with SLURM:

### First Time Setup

```bash
# Install Python 3.11 (if not available)
cd /storage/ukp/work/ibrahim1
wget https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz
tar -xzf Python-3.11.8.tgz
cd Python-3.11.8
./configure --prefix=/storage/ukp/work/ibrahim1/python3.11 --enable-optimizations
make -j$(nproc)
make install

# Add to PATH
export PATH="/storage/ukp/work/ibrahim1/python3.11/bin:$PATH"

# Create virtual environment
python3.11 -m venv /storage/ukp/work/ibrahim1/python_env
source /storage/ukp/work/ibrahim1/python_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running SLURM Experiments

```bash
# Activate environment
cd /storage/ukp/work/ibrahim1
source /storage/ukp/work/ibrahim1/python_env/bin/activate

# Submit batch job
sbatch -q yolo -p yolo run.sh

# Interactive session
srun --mem=16G --cpus-per-task=4 --gres=gpu:1 --time=01:00:00 --pty bash
source /storage/ukp/work/ibrahim1/python_env/bin/activate

# Run experiments
python -m src.main -m direct -n 5 -b slurm -om distilgpt2
python -m src.main -m writer_reviewer -n 3 -b slurm -c slurm_thinking
```

### SLURM Environment Variables

```bash
# Set cache directories for efficiency
export TRANSFORMERS_CACHE=/tmp/transformers_cache_eval
export HF_HOME=/tmp/hf_cache_eval
mkdir -p $TRANSFORMERS_CACHE $HF_HOME
```

## 🔧 Configuration System

### Unified Model Configuration

The system uses a unified configuration approach supporting both baseline and collaborative methods:

**Configuration Files**:

- `src/config/model_ollama_localhost.yaml` - Local Ollama optimization
- `src/config/model_ollama_ukp.yaml` - UKP server configuration
- `src/config/model_slurm.yaml` - SLURM execution settings
- `src/config/model_slurm_thinking.yaml` - Reasoning-optimized models

### Override System

The `--override_model` parameter provides complete control:

- **With override**: All tasks use the specified model
- **Without override**: Uses task-specific optimized models
- **Supports**: Both Ollama models (`qwen2.5:7b`) and HuggingFace models (`distilgpt2`)

## 🧪 Research Features

### Collaborative Memory System

- Persistent interaction history across iterations
- Feedback categorization and severity tracking
- Convergence analysis and stopping criteria
- Theory of Mind modeling for agent interactions

### Retrieval Integration

- Multiple retrieval backends (Wikipedia, Supabase/FAISS)
- Semantic filtering for relevance optimization
- Query generation strategies for comprehensive coverage
- Retrieved passage integration and citation tracking

### Experimental Reproducibility

- Comprehensive configuration logging
- Checkpoint system for long-running experiments
- Deterministic random seeding
- Debug mode with full artifact preservation

## 📊 Analysis & Evaluation

```bash
# Post-experiment analysis
python -m src.evaluation results/ollama/writer_reviewer_N=10_T=14.10_15:30
python -m src.analysis results/ollama/writer_reviewer_N=10_T=14.10_15:30

# Cross-experiment comparison
python -m src.analysis results/ollama/direct_N=10_T=14.10_15:30 results/ollama/writer_reviewer_N=10_T=14.10_15:30
```

## 🚧 Development & Testing

```bash
# Run comprehensive test suite
python -m src.tests

# Test specific components
python test_reviewer_standalone.py

# Validate configurations
python -c "from src.config import ModelConfig; print('✅ Config system working')"

# Test SLURM engine without cluster
python -m src.main -m direct -n 1 -b slurm -om distilgpt2 --debug
```

## 📚 Research Context

This framework is designed for research in:

- **Collaborative AI Writing**: Multi-agent content generation
- **Retrieval-Augmented Generation**: Knowledge integration strategies
- **Iterative Refinement**: Feedback-driven improvement cycles
- **Theory of Mind in AI**: Agent interaction modeling
- **Content Quality Evaluation**: Comprehensive metric development

### Dataset

- **FreshWiki**: Recent Wikipedia articles for evaluation
- **Wikipedia Dump**: Large-scale knowledge retrieval corpus
- **Supabase Embeddings**: Pre-computed semantic vectors for efficient retrieval

## 🔗 Integration & Extensions

### Adding New Methods

1. Implement method in `src/methods/your_method.py`
2. Add configuration in model YAML files
3. Register in `src/main/cli_args.py`
4. Test with `python -m src.main -m your_method -n 1 -om gemma3:1b`

### Custom Retrieval Systems

1. Implement `BaseRetrievalManager` interface
2. Add configuration in `src/config/retrieval_config.yaml`
3. Register in retrieval factory

### Model Backend Extensions

1. Implement `BaseEngine` interface in `src/engines/`
2. Add backend configuration support
3. Update CLI argument parsing

---

## 📄 License & Citation

This is a research project. Please cite appropriately if used in academic work.

## 🤝 Contributing

1. Follow PEP8 standards
2. Add comprehensive docstrings
3. Test with `python -m src.main -m [method] -n 1 -om gemma3:1b -b ollama -rm supabase_faiss`
4. Ensure backward compatibility

---

**Quick Test Command**: `python -m src.main -m direct -n 1 -om gemma3:1b -b ollama -rm supabase_faiss > test.log 2>&1`
