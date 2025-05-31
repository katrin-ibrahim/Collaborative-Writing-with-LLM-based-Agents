# AI Writer Agent Framework

This project implements a flexible AI writing framework with multiple agent approaches for generating articles on various topics. It features a collaborative writing system with separate Writer and Reviewer agents that interact through a shared memory and an iterative revision protocol.

## System Architecture

Our system comprises two clearly separated agents that interact through a shared memory and an iterative revision protocol:

- **Writer**: Responsible solely for content creation. It issues retrieval queries to an external search engine, collates the top-k passages into a working context, drafts a hierarchical outline, and then expands each heading.

- **Reviewer Team**: Comprised of three agents:
  - **Fact-Checker**: Flags unsupported or inaccurate claims
  - **Structure-Advisor**: Assesses logical flow and section ordering
  - **Leader**: Merges their comments, removes duplicates, and outputs a unified review, with each comment tagged by category and severity

- **Revision Loop**: After receiving the review, the Writer revises the draft, marking each comment as addressed or contested. Convergence is the proportion of addressed comments; the loop stops when convergence exceeds 90% or after N iterations. At each turn, the Writer and Reviewer begin with a Theory of Mind prediction of the partner's priorities.



## Agent Approaches

### Baselines

1. **Direct Prompt** - Simple one-shot generation without retrieval or structure
2. **Writer Only** - Two-step outline-to-draft approach without external knowledge
3. **RAG Writer** - Knowledge-enhanced writing with retrieval for each section
4. **Writer-Reviewer** - Advanced hierarchical knowledge organization and drafting pipeline

### Collaborative Approach

The collaborative approach combines a Writer agent with a Reviewer Team to iteratively improve the draft through a revision process. This approach can be used with any of the Writer agents and includes the full Reviewer Team.

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

### Usage

```bash
# Run with default settings (RAG Writer, first topic)
python main.py

# Run with a specific approach
python main.py --approach direct_prompt
python main.py --approach writer_only
python main.py --approach rag_writer
python main.py --approach co_storm

# Run with the collaborative approach (includes review)
python main.py --approach collaborative --review

# Select a specific topic by index
python main.py --topic_index 3

# Configure the revision process
python main.py --approach collaborative --review --iterations 5 --convergence 0.8
```


## Evaluation

The system evaluates content quality using several metrics:

- **Heading Soft Recall (HSR)**: Sentence-BERT embeddings measure topic coverage by averaging the best cosine similarity between gold and generated headings.
- **Heading Entity Recall (HER)**: FLAIR NER checks what fraction of named entities in reference headings re-appear in model heading.
- **ROUGE-1/2/L**: n-gram overlap between generated and gold articles, computed per section after each revision to capture quality improvements over iterations.
- **Article Entity Recall (AER)**: FLAIR-based recall of all named entities in the reference article.

## TODO

- [ ] clean up configs (move to folder)
- [ ] make model name configurable
- [ ] text size configurable
- [ ] prompt template for writer
- [ ] evaluator prompt template
- [ ] add web search knowledge source
- [ ] explore scientific articles knowledge source
- [ ] add user goal
- [ ] testing loop (10-20 wildseek topics)
- [ ] organize utils
- [ ] add a unified logging system
- [ ] create /knowledge folder to organize kbs
- [ ] add unit tests
- [ ] add cli support
- [ ] use section drafting style (cli)


content_generation_system/
├── src/
│   ├── agents/
│   │   ├── init.py
│   │   ├── base_agent.py          # Abstract base for all agents
│   │   ├── writer/
│   │   │   ├── init.py
│   │   │   ├── writer_agent.py    # Main writer implementation
│   │   │   └── outline_generator.py
│   │   └── reviewer/              # Future: review team agents
│   │       ├── init.py
│   │       ├── fact_checker.py
│   │       ├── structure_advisor.py
│   │       └── review_leader.py
│   ├── workflows/
│   │   ├── init.py
│   │   ├── base_workflow.py       # Abstract workflow interface
│   │   ├── direct_prompting.py    # Baseline 1
│   │   ├── writer_only.py         # Baseline 2
│   │   ├── rag_writer.py          # Baseline 3
│   │   └── full_system.py         # Future: complete system
│   ├── evaluation/
│   │   ├── init.py
│   │   ├── evaluator.py           # Main evaluation orchestrator
│   │   ├── metrics/
│   │   │   ├── init.py
│   │   │   ├── heading_metrics.py # HSR, HER
│   │   │   ├── rouge_metrics.py   # ROUGE-1/2/L
│   │   │   └── entity_metrics.py  # AER
│   │   └── benchmarks/
│   │       ├── init.py
│   │       └── freshwiki_loader.py
│   ├── retrieval/
│   │   ├── init.py
│   │   ├── search_engine.py       # External search interface
│   │   └── passage_ranker.py      # Top-k passage selection
│   ├── memory/
│   │   ├── init.py
│   │   ├── shared_memory.py       # Future: agent communication
│   │   └── revision_tracker.py    # Future: iteration management
│   └── utils/
│       ├── init.py
│       ├── config.py
│       ├── logging_setup.py
│       └── data_models.py         # Pydantic models
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── experiments/
│   ├── baselines/
│   │   ├── run_direct_prompting.py
│   │   ├── run_writer_only.py
│   │   └── run_rag_writer.py
│   └── results/
├── configs/
│   ├── base_config.yaml
│   ├── baseline_configs/
│   └── model_configs/
├── requirements.txt
├── setup.py
└── README.md
