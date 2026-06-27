# 🔍 AI Knowledge Base RAG

![CI](https://github.com/SamZombie/AI-Knowlege-Base-RAG/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

A **production-style Retrieval-Augmented Generation (RAG) system** for querying government and defense AI policy documents. Built with a fully local, private stack — no OpenAI, no cloud dependencies, no cost.

> Ask natural language questions against a corpus of NIST, DoD, OMB, and GSA AI policy documents and get grounded, cited answers in seconds.

---

## Demo

**Query:**
```json
{
  "body": "What are the four core functions of the NIST AI Risk Management Framework?"
}
```

**Response:**
```json
{
  "answer": "The four core functions are GOVERN, MAP, MEASURE, and MANAGE...",
  "sources": [
    { "filepath": "NIST_AI_100-1.pdf", "title": "AI RMF 1.0", "page": 8, "author": "NIST" }
  ]
}
```

**Evaluation Dashboard:**

![Dashboard](https://img.shields.io/badge/Retrieval%20Recall-0.95-brightgreen) ![Dashboard](https://img.shields.io/badge/Answer%20Similarity-0.92-brightgreen)

---

## Architecture

```
PDF Documents
      ↓
Document Loader (PyMuPDF)
      ↓
Chunking (RecursiveCharacterTextSplitter)
      ↓
Embeddings (BAAI/bge-large-en-v1.5)
      ↓
Qdrant Vector Store
      ↓
Semantic Retrieval (top-k ANN search)
      ↓
Prompt Builder
      ↓
Mistral 7B (Ollama)
      ↓
FastAPI REST API → JSON + Citations
```

---

## Features

- **100% Local & Free** — runs entirely on your machine, no API keys or cloud costs required
- **Privacy-First** — document content never leaves your machine; ideal for sensitive government/defense documents
- **Configurable Backend** — designed to support multiple LLM backends via environment variables
- **Source Citations** — every answer includes source filename, page number, title, and author
- **Production REST API** — FastAPI with Pydantic validation, auto-generated `/docs` UI
- **Dockerized** — full stack runs with `docker compose up`
- **Evaluation Pipeline** — automated retrieval recall and LLM-as-judge answer scoring
- **Streamlit Dashboard** — visual evaluation metrics with per-question inspection

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Mistral 7B via Ollama |
| Embeddings | BAAI/bge-large-en-v1.5 (sentence-transformers) |
| Vector Store | Qdrant |
| PDF Parsing | PyMuPDF |
| Orchestration | LangChain |
| API | FastAPI + Uvicorn |
| Evaluation | Custom LLM-as-Judge pipeline |
| Dashboard | Streamlit |
| Containerization | Docker + Docker Compose |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Ollama](https://ollama.com/download)
- Python 3.11+

### Option A — Fully Local (Free)

```bash
# 1. Clone the repo
git clone https://github.com/SamZombie/AI-Knowlege-Base-RAG.git
cd AI-Knowlege-Base-RAG

# 2. Pull the LLM
ollama pull mistral

# 3. Configure environment
cp .env.example .env

# 4. Start services
docker compose up --build

# 5. Ingest your documents (add PDFs to /docs first)
python app/ingest.py

# 6. Query the API
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -d '{"body": "What are NIST recommendations for AI governance?"}'
```

---

## Project Structure

```
ai-knowledge-base-rag/
├── app/
│   ├── main.py          # FastAPI entrypoint + endpoints
│   ├── query.py         # Embedding, retrieval, generation pipeline
│   └── ingest.py        # PDF loading, chunking, vector storage
├── tests/
│   ├── evaluate.py      # End-to-end evaluation pipeline
│   ├── dashboard.py     # Streamlit evaluation dashboard
│   └── test_questions.json  # 10 ground-truth Q&A pairs
├── docs/                # PDF corpus (not committed)
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## Evaluation Results

Evaluated against 10 ground-truth Q&A pairs covering NIST, DoD, OMB, and GSA AI policy documents.

| Metric | Score |
|---|---|
| Retrieval Recall | **0.95** |
| Answer Similarity (LLM-as-Judge) | **0.92** |

Run the evaluation yourself:

```bash
# Make sure docker compose is running first
python tests/evaluate.py

# View the dashboard
streamlit run tests/dashboard.py
```

---

## Configuration

All configuration is managed via environment variables. Copy `.env.example` to `.env` to get started.

| Variable | Description | Default |
|---|---|---|
| `LLM_BACKEND` | LLM backend to use (`ollama`) | `ollama` |
| `OLLAMA_MODEL` | Ollama model name | `mistral` |
| `OLLAMA_BASE_URL` | Ollama host URL (use `host.docker.internal` in Docker) | `http://localhost:11434` |
| `QDRANT_HOST` | Qdrant hostname (use `qdrant` in Docker) | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `BAAI/bge-large-en-v1.5` |
| `VECTOR_DIMENSIONS` | Embedding output dimensions | `1024` |
| `DOCS_PATH` | Path to PDF corpus directory | `docs` |
| `CHUNK_SIZE` | Document chunk character size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap character size | `200` |
| `TOP_K_RESULTS` | Number of chunks retrieved per query | `10` |
| `API_HOST` | API host for evaluation client | `localhost` |
| `API_PORT` | API port for evaluation client | `8000` |
| `APP_HOST` | Uvicorn bind host | `0.0.0.0` |
| `APP_PORT` | Uvicorn bind port | `8000` |

---

## Sample Documents

The `/docs` directory is not committed. Recommended public domain sources:

- [NIST AI RMF 1.0](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf)
- [NIST AI RMF Playbook](https://airc.nist.gov/Docs/2)
- [OMB M-25-21](https://www.whitehouse.gov/wp-content/uploads/2025/04/m-25-21.pdf)
- [DoD Responsible AI Strategy](https://www.defense.gov/News/Releases/Release/Article/2908233/)

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.