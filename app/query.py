import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from ollama import generate, GenerateResponse

load_dotenv()
top_k_results = os.getenv("TOP_K_RESULTS")
embedding_model = os.getenv("EMBEDDING_MODEL")
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_post = os.getenv("QDRANT_PORT")
ollama_model = os.getenv("OLLAMA_MODEL")

def embed_query(query : str) -> list[float]:
    hf = HuggingFaceEmbeddings(model_name= embedding_model)
    return hf.embed_query(query)

def search_qdrant(query_vector : list[float], collection_name : str = "default"):
    client = QdrantClient(host=qdrant_host, port=qdrant_post)
    if not client.collection_exists(collection_name= collection_name):
        raise ValueError("Collection name does not exist.")
    return client.query_points(
        collection_name= collection_name,
        query= query_vector,
        limit= int(top_k_results)
    )

def generate_prompt(user_query : str, context : list[str]) -> str:
    context_str = "\n\n".join(context)
    return f"""You are a helpful assistant. Answer the question based only on the context provided below. If the answer is not in the context, say "I don't know."
    
    Context:
    {context_str} 
    
    Question:
    {user_query} 
    
    Answer:"""

def generate_answer(query : str, collection_name : str = "default") -> GenerateResponse:
    query_vector = embed_query(query)
    context = search_qdrant(query_vector, collection_name)
    context_content = [r.payload["content"] for r in context.points]
    prompt = generate_prompt(query, context_content)
    response = generate(
        model= ollama_model,
        prompt= prompt
    )
    return response
    
answer = generate_answer("How should organizations manage AI risk according to NIST?")
print(answer.response)