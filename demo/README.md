# om-memory Demo Showcase

A polished demo for showcasing **om-memory** — human-like observational memory for AI agents.

## Quick Start

```bash
cd demo
pip install -r requirements.txt
streamlit run app.py
```

## Contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit demo: RAG vs om-memory live comparison, cost dashboard, observability |
| `om_memory_guide.ipynb` | Interactive Jupyter notebook tutorial |
| `knowledge_base.txt` | Sample company knowledge base for RAG simulation |

## What This Demonstrates

1. **Live Comparison** — Side-by-side: Traditional RAG (ChromaDB + full chat history) vs om-memory (ChromaDB + compressed observational context)
2. **Cost Savings** — Real token usage metrics showing 60-80% reduction in context window tokens
3. **Observability** — See exactly what om-memory extracts, compresses, and retains
