import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("CHUNK_SIZE", "1000")
os.environ.setdefault("CHUNK_OVERLAP", "200")
os.environ.setdefault("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
os.environ.setdefault("VECTOR_DIMENSIONS", "1024")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("TOP_K_RESULTS", "10")
os.environ.setdefault("OLLAMA_MODEL", "mistral")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("DOCS_PATH", "docs")
