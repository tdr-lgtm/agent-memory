# agent-memory

Memory techniques for LLM agents, implemented one technique per file with LangChain and LangGraph.

An LLM is stateless: every call starts from zero. This repo covers the ways to give an agent memory, from re-sending the chat history to long-term stores, memory types, update and decay strategies, memory poisoning, and evals. Each file is self-contained, runnable, and prints its results (token counts, retrieval scores, pass rates) so the trade-offs are visible.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tdr-lgtm/agent-memory.git
cd agent-memory
uv sync
cp .env.example .env
```

Fill in `.env`:

```dotenv
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY="your-key-here"
LLM_MODEL=openai/gpt-oss-120b
```

Any OpenAI-compatible endpoint works (OpenRouter, OpenAI, Groq, local servers).

Run a lesson:

```bash
uv run lessons/01_stateless_model.py
```

## Lessons

Each file has notes at the top describing the technique and its trade-offs.

### Foundations

| # | File | Covers |
|---|---|---|
| 01 | `01_stateless_model.py` | Stateless LLMs; memory as re-injected context |
| 02 | `02_memory_map.py` | Working, episodic, semantic and procedural memory in one prompt |

### Short-term memory

| # | File | Covers |
|---|---|---|
| 03 | `03_buffer_memory.py` | Full conversation buffer and token growth |
| 04 | `04_sliding_window.py` | Last-N message window |
| 05 | `05_summary_memory.py` | Progressive abstractive summarization |
| 06 | `06_summary_buffer.py` | Raw recent messages + summarized history |
| 07 | `07_checkpointer_threads.py` | LangGraph checkpointer and thread isolation |
| 08 | `08_trim_by_tokens.py` | Token-budget trimming with `trim_messages` |
| 09 | `09_summarize_node.py` | Summarize node with `RemoveMessage` |

### Long-term memory

| # | File | Covers |
|---|---|---|
| 10 | `10_vector_memory.py` | Embedding-based retrieval, built with numpy |
| 11 | `11_retrieval_scoring.py` | Relevance + recency + importance; two-stage retrieval |
| 12 | `12_profile_memory.py` | Structured profile extraction and merge |
| 13 | `13_langgraph_store.py` | Cross-thread memory with LangGraph Store and namespaces |

### Memory types

| # | File | Covers |
|---|---|---|
| 14 | `14_episodic_memory.py` | Dated session summaries with time-range recall |
| 15 | `15_semantic_memory.py` | Consolidating episodes into facts and patterns |
| 16 | `16_procedural_memory.py` | Versioned, self-rewriting system instructions |
| 17 | `17_self_reflection.py` | Test-driven reflection and lesson transfer (Reflexion) |

### Memory management

| # | File | Covers |
|---|---|---|
| 18 | `18_hot_path_vs_background.py` | Synchronous vs background memory writes |
| 19 | `19_memory_routing.py` | Routing queries to the right memory store |
| 20 | `20_conflicts_and_updates.py` | ADD / UPDATE / DELETE / NOOP with history |
| 21 | `21_forgetting_and_decay.py` | Ebbinghaus decay with use-based reinforcement |

### Production

| # | File | Covers |
|---|---|---|
| 22 | `22_memory_poisoning.py` | Prompt injection persisted to memory; write-time guards |
| 23 | `23_evaluating_memory.py` | Repeated-run evals against a no-memory baseline |

## Problems and fixes

| Problem | Fix | Lessons |
|---|---|---|
| Model forgets everything | Re-send history | 01 |
| Cost grows with chat length | Windowing, summaries, token trimming | 04–09 |
| Memory lost between chats | Per-user store | 13 |
| Irrelevant memories retrieved | Two-stage retrieval, routing | 11, 19 |
| Old and new facts conflict | Profile merge, update step | 12, 20 |
| History overwritten | Episodic log, update history | 14, 20 |
| Agent knows a preference but ignores it | Procedural rules instead of facts | 16, 23 |
| Memory fills with noise | Consolidation, decay | 15, 21 |
| Injected instructions saved | Write-time guard, memory as data | 22 |
| No proof memory helps | Evals over repeated runs | 23 |

## Structure

```
agent-memory/
├── src/agent_memory/
│   └── config.py        # get_model(), reads LLM_* from .env
├── lessons/             # 01–23, one technique per file
├── .env.example
└── pyproject.toml
```

## References

- Park et al., 2023. *Generative Agents: Interactive Simulacra of Human Behavior*
- Shinn et al., 2023. *Reflexion: Language Agents with Verbal Reinforcement Learning*
- Ebbinghaus, 1885. *Memory: A Contribution to Experimental Psychology*