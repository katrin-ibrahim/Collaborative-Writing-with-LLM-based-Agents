# AI Writer Agent Framework

A research framework for collaborative AI writing using Writer-Reviewer agents with Theory of Mind (ToM) integration. Implements iterative drafting, critique, and revision cycles alongside standard baselines to simulate editorial processes.

## Supported Methods

**Baselines:**

- `direct`: Direct prompting (Zero-shot)
- `rag`: Standard Retrieval-Augmented Generation
- `storm`: Retrieval-driven outline generation and writing

**Collaborative Agents:**

- `writer`: Single agent with iterative research and drafting
- `writer_reviewer`: Two-agent collaborative loop (Draft → Review → Revise)
- `writer_reviewer_tom`: Collaborative loop with Theory of Mind modeling

## Quick Start

```bash
# 1. Clone repository
git clone [https://github.com/katrin-ibrahim/Collaborative-Writing-with-LLM-based-Agents.git](https://github.com/katrin-ibrahim/Collaborative-Writing-with-LLM-based-Agents.git)
cd Collaborative-Writing-with-LLM-based-Agents

# 2. Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Execution

# Run collaborative agents (Writer + Reviewer)

```bash
python -m src.main -m writer_reviewer -n 5 -b ollama

# Run baselines (Direct, RAG, STORM)
python -m src.main -m direct rag storm -n 5 -b ollama
```

## Key Arguments:

- -m, --methods: Methods to run (writer_reviewer, writer, direct, rag, storm)
- -n, --num_topics: Number of topics to generate
- -b, --backend: Backend execution ('ollama' or 'slurm')
- -om, --override_model: Force specific model (e.g., 'qwen2.5:32b')

# Evaluate a specific run directory

```bash
python -m src.evaluation results/ollama/writer_reviewer_N=5_T=14.10_15:30
```

## Output Structure

Results are saved to results/{backend}/{experiment_name}/:

- articles/: Generated articles (.md) and metadata (.json)
- results.json: Aggregate metrics and configuration
- memory/: Memory log for writer, or writer_reviewer executions
