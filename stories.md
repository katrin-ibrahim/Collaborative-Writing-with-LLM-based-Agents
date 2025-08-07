### No pages matching the search query

- ID: No pages matching the search query
- Type: bug
- Priority: medium
- Description: For some queries no wiki pages are retrieved, specifically for storm
  try adjusting the configuration

### Create Main Entry Points

- ID: Create Main Entry Points
- Type: technical-task
- Priority: medium
- Description: Create CLI entry points for writer-only and writer-reviewer workflows.

**Acceptance Criteria**

- Writer CLI
- Writer-Reviewer CLI
- Argument parsing
- Works with experiment framework
- Outputs compatible with evaluation system

**Dependencies**
WR-004, WR-005

### Writer-Reviewer Integration Testing

- ID: Writer-Reviewer Integration Testing
- Type: test
- Priority: medium
- Description: Test full writer-reviewer collaboration loop.

**Acceptance Criteria**

- End-to-end test
- Convergence accuracy
- Feedback loop stability
- Baseline comparison
- Evaluate quality: HSR, HER, ROUGE, AER

**Dependencies**
WR-005, WR-006, WR-007

### Writer Agent Standalone Testing

- ID: Writer Agent Standalone Testing
- Type: test
- Priority: medium
- Description:
  Create test suite for Writer agent in isolation.

**Acceptance Criteria**

- Unit tests for nodes
- Full workflow test
- Mock LLM responses
- Performance benchmarks
- Error/recovery tests

**Dependencies**
WR-004

### Theory of Mind Integration

- ID: Theory of Mind Integration
- Type: feature
- Priority: medium
- Description: Add Theory of Mind prompts to simulate agents predicting each other’s goals.

**Acceptance Criteria**

- Simple ToM prompts
- Injected at start of interaction
- Context passing between agents
- A/B test ToM vs non-ToM
- Configurable complexity levels

**Dependencies**
WR-004, WR-005

### Implement Shared Memory for Collaboration

- ID: Implement Shared Memory for Collaboration
- Type: technical-task
- Priority: medium
- Description: Add SharedMemory class to support writer-reviewer iteration tracking and convergence.

**Acceptance Criteria**

- Track revisions, feedback, responses
- Support "addressed" vs "contested"
- Calculate convergence (90%)
- Integrate with StateManager or as standalone
- Thread-safe design

**Technical Details**
Handles max iterations + convergence checks
Choose integration vs isolation

### Create Reviewer Agent Architecture

- ID: Create Reviewer Agent Architecture
- Type: feature
- Priority: medium
- Description: Implement ReviewerAgent with fact-checking, structure analysis, and feedback generation.

**Acceptance Criteria**

- Fact-checking via tools
- Analyze structure and content flow
- Feedback with categories + severity
- Structured comments
- LangGraph-compatible
- Plug into evaluation framework

**Dependencies**
WR-002 (LangGraph Tools)

### Implement 3-Node Writer LangGraph Workflow

- ID: Implement 3-Node Writer LangGraph Workflow
- Type: feature
- Priority: medium
- Description:
  Create simplified 3-node Writer agent using LangGraph and autonomous tool selection.

**Acceptance Criteria**

- Implement nodes: research, plan_outline, write_content
- Add conditional research-more loop
- Output matches baseline format
- Integrate with state/output management

**Technical Details**
Replace 6-node graph with 3-node
Each node uses LLM + tools
State format compatible with baseline

### Verify KB/RM Integration

- ID: Verify KB/RM Integration
- Type: technical-task
- Priority: medium
- Description: Ensure RetrievalManager (RM) and KnowledgeBase (KB) interoperate correctly with clean data flow and format handling.

**Acceptance Criteria**

- RM retrieves data
- KB organizes results
- Semantic search works
- Format handling: strings, dicts
- Caching functional
- Errors handled gracefully

**Technical Details**
Test: `RM.search() → KB.organize() → KB.get_content()`
Verify deduplication
Support RAG and STORM outputs

### Convert Toolkits to LangGraph Tools

- ID: Convert Toolkits to LangGraph Tools
- Type: technical-task
- Priority: medium
- Description: Convert toolkit methods to proper LangGraph `@tool` decorators for LLM-driven decision making.

**Acceptance Criteria**

- Convert SearchToolkit, ContentToolkit, KnowledgeToolkit, EvaluationToolkit methods to `@tool`
- Follow LangGraph naming conventions
- Return structured data suitable for LLMs

**Technical Details**
Replace calls like `self.toolkit.search.search_web()`
Add proper docstrings
Enable dynamic tool selection

### Fix BaseAgent Constructor and Imports

- ID: Fix BaseAgent Constructor and Imports
- Type: bug
- Priority: medium
- Description: BaseAgent has duplicate `__init__` methods and a missing `OllamaClient` import, causing initialization failures.

**Acceptance Criteria**

- Remove duplicate `__init__` method
- Add `from src.utils.ollama_client import OllamaClient` import
- Ensure constructor has a single valid definition
- BaseAgent can be instantiated without errors

**Technical Details**
File: `src/agents/base_agent.py`
Nested constructors currently cause syntax errors
All agents inherit from BaseAgent — current bug breaks them all
