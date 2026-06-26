import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()
chunk_size = os.getenv("CHUNK_SIZE")
chunk_overlap = os.getenv("CHUNK_OVERLAP")
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_post = os.getenv("QDRANT_PORT")
docs_path = os.getenv("DOCS_PATH")
embedding_model = os.getenv("EMBEDDING_MODEL")
vector_dimensions = os.getenv("VECTOR_DIMENSIONS")

def load_documents() -> list[Document]:
    parent_dir = Path(__file__).resolve().parent.parent
    docs_dir = os.path.join(parent_dir, docs_path)
    docs = []
    for file in os.listdir(docs_dir):
        if not file.casefold().endswith(".pdf"):
            continue
        path = os.path.join(docs_dir, file)
        loader = PyMuPDFLoader(file_path=path)
        doc_loaded = loader.load()
        docs.extend(doc_loaded)
    return docs

def split_documents(docs : list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= int(chunk_size),
        chunk_overlap= int(chunk_overlap)
    )
    return splitter.split_documents(docs)

def store_embeddings(docs : list[Document], collection_name : str = "default") -> bool:
    client = QdrantClient(host=qdrant_host, port=qdrant_post)
    hf = HuggingFaceEmbeddings(model_name= embedding_model)
    vector = VectorParams(size=vector_dimensions, distance=Distance.COSINE)
    embeded_docs = hf.embed_documents([doc.page_content for doc in docs])

    if not client.collection_exists(collection_name= collection_name):
        client.create_collection(
            collection_name= collection_name,
            vectors_config= vector
        )
    points = []
    for idx, (embed, doc) in enumerate(zip(embeded_docs, docs)):
        payload = {"filename": doc.metadata["source"],
                   "page": doc.metadata["page"],
                    "title": doc.metadata.get("title", "Unknown"),
                   "author": doc.metadata.get("author", "Unknown"),
                   "content": doc.page_content
                   }
        point = PointStruct(
            id= idx,
            vector= embed,
            payload= payload
        )
        points.append(point)
    client.upsert(
        collection_name= collection_name,
        points= points
    )
    return True

# docs = load_documents()
# chunks = split_documents(docs)
# store_embeddings(chunks)