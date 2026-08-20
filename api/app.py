# Import lru_cache so expensive objects can be created once and reused.
# This matters because loading embedding models or rerankers can be slow.
from functools import lru_cache

# FastAPI creates the web/API application.
# BaseModel defines the expected structure of request/response data.
from fastapi import FastAPI
from pydantic import BaseModel

# These are possible LLM backends.
# The API can choose between them based on the request.
from llm.mock_llm import MockLLM
from llm.ollama_llm import OllamaLLM
from llm.openai_llm import OpenAILLM

# RAGChain is the orchestrator:
# retriever -> optional reranker -> prompt builder -> LLM.
from llm.rag_chain import RAGChain

# CrossEncoderReranker improves the initial retrieval order.
from reranking.cross_encoder import CrossEncoderReranker

# Retriever searches the FAISS vector index.
from retrieval.query import Retriever


# Create the FastAPI app object.
# This object holds all API routes/endpoints.
app = FastAPI(title="Production RAG System")


# Defines the JSON body the user sends to POST /query.
# FastAPI validates incoming requests using this model.
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    retrieve_k: int = 20
    min_score: float = 0.3
    llm_provider: str = "mock"
    use_reranker: bool = True


# Defines the JSON response returned by POST /query.
# This keeps API output predictable.
class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]


# Creates and caches the retriever.
# Without caching, the FAISS index/model could reload on every request.
@lru_cache(maxsize=1)
def get_retriever():
    retriever = Retriever()
    return retriever


# Creates and caches the reranker.
# Cross-encoder models are expensive to load, so we reuse one instance.
@lru_cache(maxsize=1)
def get_reranker():
    reranker = CrossEncoderReranker()
    return reranker


# Selects which LLM implementation to use.
# All LLM classes expose the same generate(messages) method.
def build_llm(provider_name: str):
    if provider_name == "mock":
        llm = MockLLM()

    elif provider_name == "ollama":
        llm = OllamaLLM()

    elif provider_name == "openai":
        llm = OpenAILLM()

    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    return llm


# Builds the complete RAG pipeline for this request.
# It wires together retriever, LLM, and optional reranker.
def build_rag_chain(llm_provider: str, use_reranker: bool):
    retriever = get_retriever()
    llm = build_llm(llm_provider)

    if use_reranker:
        reranker = get_reranker()
    else:
        reranker = None

    rag_chain = RAGChain(
        retriever=retriever,
        llm=llm,
        reranker=reranker,
    )

    return rag_chain


# Simple health endpoint.
# Used to check whether the API server is alive.
@app.get("/health")
def health_check():
    response = {
        "status": "ok",
    }

    return response


# Main RAG endpoint.
# This is where an outside user sends a question.
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    # Extract request fields into local variables.
    # This makes debugging easier because each value is visible separately.
    query = request.query
    top_k = request.top_k
    retrieve_k = request.retrieve_k
    min_score = request.min_score
    llm_provider = request.llm_provider
    use_reranker = request.use_reranker

    # Build the RAG pipeline using the requested settings.
    rag_chain = build_rag_chain(
        llm_provider=llm_provider,
        use_reranker=use_reranker,
    )

    # Run the actual RAG pipeline.
    result = rag_chain.answer(
        query=query,
        top_k=top_k,
        retrieve_k=retrieve_k,
        min_score=min_score,
    )

    # Return structured JSON response.
    return result