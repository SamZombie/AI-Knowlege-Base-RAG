import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from app.query import generate_answer, search_qdrant, embed_query

app = FastAPI()

class Query(BaseModel):
    """Represents an incoming user query request body."""
    body: str

class Source(BaseModel):
    """Represents a single document source citation returned with an answer."""
    filepath: str
    title: str
    page: int
    author: str

class Answer(BaseModel):
    """Represents the full API response containing the generated answer and its sources."""
    answer: str
    sources: list[Source]


@app.post("/query/")
async def ask_query(query: Query, collection_name: str = "default") -> Answer:
    """
    Handles POST /query/ requests by running the full RAG pipeline.

    Embeds the user query, retrieves the most semantically similar document
    chunks from Qdrant, generates a grounded answer using the local Ollama
    model, and returns the answer alongside source citations.

    Args:
        query (Query): The request body containing the user's question.
        collection_name (str): The Qdrant collection to search against.
                               Defaults to "default".

    Returns:
        Answer: A response object containing the generated answer string and
        a list of Source objects with filepath, title, author, and page number
        for each retrieved chunk.
    """
    query_vector = embed_query(query.body)
    context = search_qdrant(query_vector, collection_name)
    context_content = [r.payload["content"] for r in context.points]
    answer_str = generate_answer(query.body, context_content).response
    sources = []
    seen = set()
    
    for r in context.points:
        id = (r.payload["filename"], r.payload["page"])

        if id not in seen:
            seen.add(id)
            sources.append( Source(
            filepath= r.payload["filename"],
            title= r.payload["title"],
            page= r.payload["page"],
            author= r.payload["author"]
            ))
        
    answer = Answer(
        answer= answer_str,
        sources= sources
    )
    
    return answer