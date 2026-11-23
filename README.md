# langgraph-rag-agent-ollama-langfuse

A private document Q&A agent that runs entirely on local LLMs (Ollama / vLLM). Your documents never leave your machine, and every answer will cite its source.

**Stack:** Python · Ollama / vLLM · RAG (Qdrant) · LangGraph · Pydantic · Ragas · Langfuse · FastAPI · Docker · GitHub Actions

> Status: **Phase 1 – local model client.** Built step by step; see the roadmap below.

## Why

Many teams want LLMs on sensitive data (contracts, policies, internal manuals) but cannot send it to a cloud API. This project explores how to build that reliably: local inference, retrieval with citations, structured outputs, guardrails and evaluation.

## Quick start

```bash
# 1. Install and start Ollama: https://ollama.com
ollama pull llama3.2

# 2. Set up Python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Ask a question
python -m private_docs_rag.cli "What is retrieval-augmented generation?"

# 4. Run tests (no model needed)
pytest
```

Any OpenAI-compatible server works (vLLM, llama.cpp, LM Studio). Set `LLM_BASE_URL` and `LLM_MODEL` (see `.env.example`).

## Roadmap

- [x] Phase 1 – Local model client with tests
- [ ] Phase 2 – RAG: ingest PDFs, embed locally, retrieve with page citations, trace every step with self-hosted Langfuse
- [ ] Phase 3 – LangGraph agent: retrieve → grade → generate → validate, with retries, tools and structured outputs (Pydantic)
- [ ] Phase 4 – Evaluation set, hallucination guardrails, PII masking
- [ ] Phase 5 – FastAPI, Docker Compose (app + Qdrant + Ollama + Langfuse), CI, prompt versioning and scores in Langfuse

## License

This project is licensed for non-commercial use only.  
See the LICENSE file for details.
