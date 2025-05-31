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

# Run with a specific approach
python main.py --method direct --num_topics 5
python main.py --method writer_only --num_topics 5
python main.py --method rag --num_topics 5

# Run all baselines
python main.py --method all --num_topics 5


```


## Evaluation

The system evaluates content quality using several metrics:

- **Heading Soft Recall (HSR)**: Sentence-BERT embeddings measure topic coverage by averaging the best cosine similarity between gold and generated headings.
- **Heading Entity Recall (HER)**: FLAIR NER checks what fraction of named entities in reference headings re-appear in model heading.
- **ROUGE-1/2/L**: n-gram overlap between generated and gold articles, computed per section after each revision to capture quality improvements over iterations.
- **Article Entity Recall (AER)**: FLAIR-based recall of all named entities in the reference article.

## TODO

- [ ] make model name configurable
- [ ] text size configurable
- [ ] prompt template for writer
- [ ] add web search knowledge source
- [x] organize utils
- [x] add a unified logging system
- [x] create /knowledge folder to organize kbs
- [ ] add unit tests
- [x] add cli support
- [ ] use section drafting style (cli)

