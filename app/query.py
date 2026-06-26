import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from ollama import GenerateResponse, Client

load_dotenv()
top_k_results = os.getenv("TOP_K_RESULTS")
embedding_model = os.getenv("EMBEDDING_MODEL")
qdrant_host = os.getenv("QDRANT_HOST")
qdrant_post = os.getenv("QDRANT_PORT")
ollama_model = os.getenv("OLLAMA_MODEL")
ollama_base_url = os.getenv("OLLAMA_BASE_URL") 

qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_post)
ollama_client = Client(host= ollama_base_url)


def embed_query(query: str) -> list[float]:
    """
    Converts a user query string into a vector embedding.

    Initializes a HuggingFace embedding model using the model name from
    environment variables and embeds the query as a single string. Must use
    the same embedding model as was used during document ingestion to ensure
    vector space consistency during similarity search.

    Args:
        query (str): The user's natural language query string.

    Returns:
        list[float]: A vector embedding representing the semantic meaning of the query.
    """
    hf = HuggingFaceEmbeddings(model_name= embedding_model)
    return hf.embed_query(query)


def search_qdrant(query_vector: list[float], collection_name: str = "default"):
    """
    Searches a Qdrant collection for the most semantically similar document chunks.

    Initializes a Qdrant client using host and port from environment variables
    and performs a nearest neighbor search against the specified collection.
    Returns the top-k most similar chunks as determined by the TOP_K_RESULTS
    environment variable.

    Args:
        query_vector (list[float]): The embedded representation of the user query.
        collection_name (str): The name of the Qdrant collection to search.
                               Defaults to "default".

    Raises:
        ValueError: If the specified collection does not exist in Qdrant.

    Returns:
        QueryResponse: A Qdrant response object containing the top-k matching points
        with their vectors, scores, and payloads.
    """
    if not qdrant_client.collection_exists(collection_name= collection_name):
        raise ValueError("Collection name does not exist.")
    return qdrant_client.query_points(
        collection_name= collection_name,
        query= query_vector,
        limit= int(top_k_results)
    )


def generate_prompt(user_query: str, context: list[str]) -> str:
    """
    Builds a RAG prompt by combining retrieved context chunks with the user query.

    Joins the context chunks with double newlines for clear separation and
    wraps them in a structured prompt that instructs the model to answer only
    from the provided context. If the answer cannot be found in the context,
    the model is instructed to respond with "I don't know."

    Args:
        user_query (str): The user's original natural language question.
        context (list[str]): A list of relevant document chunk strings retrieved
                             from Qdrant.

    Returns:
        str: A fully formatted prompt string ready to be sent to the LLM.
    """    
    context_str = "\n\n".join(context)
    return f"""You are a helpful assistant. Answer the question based only on the context provided below. If the answer is not in the context, say "I don't know."
    
    Context:
    {context_str} 
    
    Question:
    {user_query} 
    
    Answer:"""


def generate_answer(query: str, context_content: list[str]) -> GenerateResponse:
    """
    Generates a natural language answer from a query and retrieved context.

    Builds a RAG prompt from the query and context chunks, then sends it to
    the configured Ollama model for generation. The model is instructed to
    answer only from the provided context to minimize hallucination.

    Args:
        query (str): The user's natural language question.
        context_content (list[str]): A list of relevant document chunk strings
                                     retrieved from Qdrant to use as context.

    Returns:
        GenerateResponse: The Ollama response object. Access the generated text
        via .response on the returned object.
    """
    prompt = generate_prompt(query, context_content)
    response = ollama_client.generate(
        model= ollama_model,
        prompt= prompt
    )
    return response