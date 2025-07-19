# Local STORM Runner

This directory contains a local STORM implementation that uses the same local models as the direct prompting baseline, allowing you to develop and test STORM locally before SLURM deployment.

## Quick Start

### 1. Install Dependencies

```bash
# Install STORM package
pip install knowledge-storm

# Install DSPy if not already installed
pip install dspy-ai
```

### 2. Test STORM Locally

```bash
# Run a quick test
cd src/local
python test_storm.py
```

### 3. Run STORM on Topics

```bash
# Run STORM on 3 topics
python -m local --methods storm --topic_limit 3

# Run both direct prompting and STORM
python -m local --methods direct_prompting storm --topic_limit 5

# Resume an existing experiment
python -m local --methods storm --resume --experiment_name my_experiment
```

## Features

### 🚀 **Same Models as Direct Prompting**

- Uses your local Qwen models (same as `LocalBaselineRunner`)
- No need for separate model setup or Ollama
- Consistent with your existing local infrastructure

### ⚡ **Optimized for Local Development**

- Reduced STORM parameters for faster testing:
  - `max_conv_turn`: 2 (vs 4 in baselines)
  - `max_perspective`: 2 (vs 4 in baselines)
  - `search_top_k`: 3 (vs 5 in baselines)
  - `max_thread_num`: 1 (for stability)

### 🔧 **SLURM-Ready Architecture**

- Same structure as baselines STORM implementation
- Easy to adapt for SLURM deployment
- Compatible output format with existing evaluation pipeline

## Architecture

```
LocalSTORMRunner
├── LocalLiteLLMWrapper (adapts local models to STORM)
├── STORMWikiRunner (from knowledge-storm package)
├── WikipediaSearchRM (search/retrieval)
└── OutputManager (same as direct prompting)
```

## Configuration

The STORM runner uses different temperature settings for different components:

- **Conversation Simulation**: temp=0.8, max_tokens=512
- **Question Asking**: temp=0.7, max_tokens=256
- **Outline Generation**: temp=0.5, max_tokens=1024
- **Article Writing**: temp=0.3, max_tokens=1024
- **Article Polishing**: temp=0.2, max_tokens=1024

## Output Structure

Same as baselines - articles saved to `articles/storm_{topic}.md` with metadata in `results.json`:

```json
{
  "results": {
    "Topic Name": {
      "storm": {
        "title": "Topic Name",
        "word_count": 1250,
        "generation_time": 45.2,
        "method": "storm",
        "local_model": true,
        "storm_config": {...}
      }
    }
  }
}
```

## Development Workflow

1. **Develop Locally**: Test STORM with `test_storm.py`
2. **Run Small Batches**: Use `--topic_limit 3` for quick tests
3. **Validate Output**: Check generated articles and metadata
4. **Deploy to SLURM**: Adapt for your SLURM environment

## Troubleshooting

### Import Errors

```bash
# Missing knowledge-storm
pip install knowledge-storm

# Missing dspy
pip install dspy-ai
```

### Model Issues

- Ensure your local models are in the `models/` directory
- Same requirements as direct prompting
- Check GPU availability and memory

### Performance

- STORM is much slower than direct prompting (multi-stage pipeline)
- Start with `topic_limit=1` for initial testing
- Use the test script for quick validation

## Next Steps

Once local development is complete:

1. Adapt the STORM runner for your SLURM environment
2. Update model paths and resource allocation
3. Test with larger topic batches
4. Compare results with Ollama STORM baseline
