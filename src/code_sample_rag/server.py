# Imports
# Python
from typing import Optional

# Codebase
from code_sample_rag import setup, retrieval, generate

# 3rd-party
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

### CONSTANTS ###
RAG_RES_KEY = "rag_resources"

# Expected input format for /query RAG API
class RAGQueryRequest(BaseModel):
    request: str
    session_id: Optional[str] = None # Keep track of sessions for multi-turn

# Output format for /query RAG API
class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]

session_to_interaction_map = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_resources = setup.RAGResources()
    app.state.resources = rag_resources
    try:
        yield
    finally:
        app.state.resources.tear_down()
    

app = FastAPI(lifespan=lifespan)

# POST: Query RAG endpoint
# Note that in practice, one user will call this at a time, and I 
# am not using aync Google GenAI libraries under the hood, so this
# func stays synchronous
@app.post("/query")
def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    query = request.query
    try:
        # TODO: Use prior chats for semantic retrieval
        retrieved_results = retrieve_context(
            query=query,
            embed_model=app.state.resources.em,
            index=resources.index,
            doc_metadata=resources.doc_metadata,
            top_k=resources.config["retrieval"]["num_results"]
        )

        # Use previousinteraction id for session if session was passed
        prev_inter_id = None
        if request.session_id:
            prev_inter_id = session_to_interaction_map.get(request.session_id)

        answer, inter_id = generate.answer(
            query=query,
            config=resources.config,
            retrieved_results=retrieved_results,
            llm=resources.llm,
            prev_inter_id=prev_inter_id
        )

        # Store interaction id for conversation that just occurred to session map if session was passed
        if request.session_id:
            session_to_interaction_map[request.session_id] = inter_id

        # TODO: Format retrieved results
        return {"answer": answer, "sources": retrieved_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error, hit: {e}")

# TODO: Add POST path for adding documents to data set