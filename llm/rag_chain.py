from llm.prompt_builder import build_prompt


class RAGChain:

    def __init__(self, retriever, llm, reranker=None):
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker

    def answer(self, query, top_k=5, retrieve_k=20, min_score=0.3):
        chunks = self.retriever.search(query, top_k=retrieve_k)

        if self.reranker is not None:
            chunks = self.reranker.rerank(query, chunks, top_k=top_k)
        else:
            chunks = chunks[:top_k]

        if not chunks or chunks[0].get("score", 0) < min_score:
            return {
                "query": query,
                "answer": "I don't have enough supporting information.",
                "sources": [],
                "contexts": [],
            }

        messages = build_prompt(query, chunks)
        response = self.llm.generate(messages)

        return {
            "query": query,
            "answer": response,
            "sources": [c["chunk_id"] for c in chunks],
            "contexts": [c["text"] for c in chunks],
        }
