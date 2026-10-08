from contextlib import asynccontextmanager

# Import lru_cache so expensive objects can be created once and reused.
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

# FastAPI creates the web/API application.
from fastapi import FastAPI
from pydantic import BaseModel

# These are possible LLM backends.
from llm.mock_llm import MockLLM
from llm.ollama_llm import OllamaLLM
from llm.openai_llm import OpenRouterLLM

# RAGChain is the orchestrator:
from llm.rag_chain import RAGChain

# CrossEncoderReranker improves the initial retrieval order.
from reranking.cross_encoder import CrossEncoderReranker

# Retriever searches the FAISS vector index.
from retrieval.query import Retriever

# Load the index and both models at startup, so the API accepts requests
# only when it is ready and the first query is not slower than the rest.
@asynccontextmanager
async def lifespan(app: FastAPI):
    get_retriever()
    get_reranker()
    yield


app = FastAPI(title="Production RAG System", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    retrieve_k: int = 20
    min_score: float = 0.3
    llm_provider: str = "mock"
    use_reranker: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    contexts: list[str]
    model: str
    server: str


@lru_cache(maxsize=1)
def get_retriever():
    retriever = Retriever()
    return retriever


@lru_cache(maxsize=1)
def get_reranker():
    reranker = CrossEncoderReranker()
    return reranker


def build_llm(provider_name: str):
    if provider_name == "mock":
        llm = MockLLM()

    elif provider_name == "ollama":
        llm = OllamaLLM()

    elif provider_name == "openrouter":
        llm = OpenRouterLLM()

    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    return llm


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


@app.get("/health")
def health_check():
    response = {
        "status": "ok",
    }

    return response


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    query = request.query
    top_k = request.top_k
    retrieve_k = request.retrieve_k
    min_score = request.min_score
    llm_provider = request.llm_provider
    use_reranker = request.use_reranker

    llm = build_llm(llm_provider)
    rag_chain = build_rag_chain(
        llm_provider=llm_provider,
        use_reranker=use_reranker,
    )

    result = rag_chain.answer(
        query=query,
        top_k=top_k,
        retrieve_k=retrieve_k,
        min_score=min_score,
    )

    result["model"] = llm.model
    result["server"] = llm.server

    return result