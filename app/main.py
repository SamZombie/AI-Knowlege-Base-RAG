import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from app.query import generate_answer, search_qdrant, embed_query

app = FastAPI()

class Query(BaseModel):
    body: str

class Source(BaseModel):
    filepath: str
    title: str
    page: int
    author: str

class Answer(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query/")
async def ask_query(query: Query, collection_name : str = "default") -> Answer:
    query_vector = embed_query(query.body)
    context = search_qdrant(query_vector, collection_name)
    context_content = [r.payload["content"] for r in context.points]
    answer_str = generate_answer(query.body, context_content).response
    sources = []
    
    for r in context.points:
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