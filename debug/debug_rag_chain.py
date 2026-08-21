import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm.mock_llm import MockLLM
from llm.rag_chain import RAGChain
from reranking.cross_encoder import CrossEncoderReranker
from retrieval.query import Retriever


retriever = Retriever()
reranker = CrossEncoderReranker()
llm = MockLLM()

rag_chain = RAGChain(
    retriever=retriever,
    reranker=reranker,
    llm=llm,
)

response = rag_chain.answer(
    query="What is the Manhattan Project?",
    retrieve_k=20,
    top_k=5,
)

print(response)