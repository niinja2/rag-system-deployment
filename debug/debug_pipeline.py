import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from retrieval.query import Retriever
from reranking.cross_encoder import CrossEncoderReranker
from llm.prompt_builder import build_prompt


def main():

    query = "What is the Manhattan Project?"

    retriever = Retriever()
    chunks = retriever.search(query, top_k=20)

    print("FAISS results:")

    for rank, chunk in enumerate(chunks, start=1):
        print(f"\n{rank}. ID: {chunk['chunk_id']}")
        print(f"FAISS score: {chunk['score']}")
        print(chunk["text"][:300])

    reranker = CrossEncoderReranker()
    reranked_chunks = reranker.rerank(
        query,
        chunks,
        top_k=5,
    )

    messages = build_prompt(query, reranked_chunks)

    print("\nPROMPT MESSAGES:")
    print(messages)

    print("\nRERANKED RESULTS:")

    for rank, chunk in enumerate(reranked_chunks, start=1):
        print(f"\n{rank}. ID: {chunk['chunk_id']}")
        print(f"FAISS score: {chunk['score']}")
        print(f"Reranker score: {chunk['rerank_score']}")
        print(chunk["text"][:300])


if __name__ == "__main__":
    main()
