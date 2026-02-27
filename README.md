# om-memory

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/om-memory.svg)
![PyPI Version](https://img.shields.io/pypi/v/om-memory)

**Human-like memory for AI agents. 10x cheaper than RAG. Zero vector DB needed.**

`om-memory` is the first-ever Python implementation of **Observational Memory (OM)** — a revolutionary approach to AI agent memory. Instead of embedding every message and retrieving similar ones per turn (RAG), OM continuously compresses conversation history into a dense, evolving text log of "observations" using two background agents (Observer and Reflector).

## What is Observational Memory?

Observational Memory (OM) maintains a stable text context window of the user's conversation. 
The observation log is plain text that stays stable in the context window, enabling prompt caching (75-90% token cost discount from providers like OpenAI/Anthropic). Traditional RAG injects different retrieved chunks every turn, breaking the cache and costing up to 10x more.

## Why om-memory?
- **10x Cheaper than RAG**: By leveraging prompt caching on a stable context window.
- **Zero Vector DB Needed**: Uses standard storage backends (SQLite, Postgres, etc.) — no embeddings, no vector search.
- **Better Accuracy**: Maintains narrative continuity better than fragmented vector retrieval (Highest recorded on LongMemEval).
- **Framework-Agnostic**: Middleware pattern. `om-memory` manages context, you make your own LLM calls with LangChain, LlamaIndex, or raw Python.
- **Observable by Default**: Full event tracking, metrics, and Streamlit dashboard integration.

## Quick Start

`om-memory` provides sensible defaults out of the box (SQLite storage, OpenAI `gpt-4o-mini` for background compression).

```python
import asyncio
from om_memory import ObservationalMemory

async def main():
    # 1. Initialize OM (Zero config needed, uses SQLite + OPENAI_API_KEY)
    om = ObservationalMemory()
    
    # 2. Get context for a user thread
    thread_id = "user_123"
    context = await om.aget_context(thread_id=thread_id)
    
    # 3. Build your prompt and call YOUR LLM
    prompt = f"System: You are a helpful assistant.\n{context}\nUser: Hello!"
    response = "Hello! How can I help you today?" # Replace with actual LLM call
    
    # 4. Tell OM what happened so it can remember for next time
    await om.aadd_message(thread_id=thread_id, role="user", content="Hello!")
    await om.aadd_message(thread_id=thread_id, role="assistant", content=response)

if __name__ == "__main__":
    asyncio.run(main())
```

## How It Works

1. **Block 1 (Observations):** A compressed, timestamped log of facts, decisions, and preferences. Handled by the **Reflector** agent.
2. **Block 2 (Recent Messages):** The uncompressed recent turns of the conversation. Once this grows past a threshold, the **Observer** agent compresses it into Block 1.

## Installation

```bash
pip install om-memory
```

Optional dependencies:
- `pip install om-memory[anthropic]` - Anthropic provider support
- `pip install om-memory[postgres]` - PostgreSQL storage backend
- `pip install om-memory[dashboard]` - Streamlit dashboard

## Documentation

Full documentation available in the repository examples.

## License

Apache 2.0
