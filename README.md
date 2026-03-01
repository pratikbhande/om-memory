# om-memory

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/om-memory.svg)
![PyPI Version](https://img.shields.io/pypi/v/om-memory)

**Human-like memory for AI agents. Cheaper than RAG. Zero vector DB needed.**

`om-memory` is a Python implementation of **Observational Memory (OM)** — a smarter approach to AI agent memory inspired by [Mastra's OM architecture](https://mastra.ai/docs/memory/observational-memory). Instead of stuffing full conversation history into every API call, OM compresses old messages into dense observations using two background agents (Observer & Reflector).

## Benchmark Results (Real API Calls)

Tested with **gpt-4o-mini** over a 50-turn HR chatbot conversation:

| Metric | Traditional RAG | om-memory | Improvement |
|--------|----------------|-----------|-------------|
| **Total tokens** | 73,599 | 54,058 | **27% savings** |
| **Per-turn at turn 50** | 2,824 tokens | 1,559 tokens | **45% savings** |
| **Memory accuracy** | 100% (full history) | 100% (8/8 recall) | **No loss** |
| **Context growth** | Linear O(n) | Flat O(1) | **Stable** |

> RAG token usage grows linearly with every turn. om-memory stays flat — the longer the conversation, the bigger the savings.

## How It Works

```
Traditional RAG:    [System] + [KB] + [ALL Messages]        ← grows every turn
om-memory:          [System] + [KB] + [Observations] + [Last 2 msgs]  ← stays flat
```

1. **Observer**: When message history exceeds a token threshold, compresses messages into concise observations (facts, decisions, preferences)
2. **Reflector**: When observations pile up, merges and prunes them — like a garbage collector for memory
3. **Context Builder**: Serves a two-block context: compressed observations + recent messages

The result: your agent remembers everything important without carrying every raw message.

## Quick Start

```python
import asyncio
from om_memory import ObservationalMemory

async def main():
    # 1. Initialize (uses SQLite + OPENAI_API_KEY by default)
    om = ObservationalMemory()
    await om.ainitialize()

    thread_id = "user_123"

    # 2. Get compressed context for your prompt
    context = await om.aget_context(thread_id)

    # 3. Use it in your LLM call
    prompt = f"You are a helpful assistant.\n{context}\nUser: Hello!"
    response = "Hello! How can I help?"  # your LLM call here

    # 4. Tell OM what happened
    await om.aadd_message(thread_id, "user", "Hello!")
    await om.aadd_message(thread_id, "assistant", response)

asyncio.run(main())
```

## Configuration

```python
from om_memory import ObservationalMemory, OMConfig

config = OMConfig(
    observer_token_threshold=300,    # compress after ~3 exchanges
    reflector_token_threshold=1500,  # GC observations early
    message_retention_count=2,       # keep last 2 messages uncompressed
    message_token_budget=200,        # token budget for recent messages
)

om = ObservationalMemory(api_key="sk-...", config=config)
```

## Installation

```bash
pip install om-memory
```

Optional extras:
```bash
pip install om-memory[postgres]     # PostgreSQL storage
pip install om-memory[anthropic]    # Anthropic provider
pip install om-memory[gemini]       # Google Gemini provider
pip install om-memory[dashboard]    # Streamlit dashboard
```

## Why Not Traditional RAG?

| | Traditional RAG | om-memory |
|---|---|---|
| **Context size** | Grows linearly with turns | Stays flat |
| **Infrastructure** | Vector DB + embeddings | SQLite (zero setup) |
| **Memory type** | Retrieves fragments | Maintains narrative |
| **Long conversations** | Hits context limit | Unlimited |
| **Cost trend** | Increases per turn | Stable per turn |

## Try It Yourself

Run the benchmark locally:
```bash
pip install om-memory openai
export OPENAI_API_KEY="sk-..."
python demo/generate_graphs.py
```

Or try the interactive [Colab notebook](demo/om_memory_colab.ipynb).

## License

Apache 2.0
