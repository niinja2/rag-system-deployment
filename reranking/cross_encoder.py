from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        # Step 1: create one query-text pair for every chunk.
        pairs = []

        for chunk in chunks:
            chunk_text = chunk["text"]
            pair = (query, chunk_text)
            pairs.append(pair)

        # Step 2: calculate one relevance score for every pair.
        scores = self.model.predict(pairs)

        # Step 3: attach each score to its corresponding chunk.
        scored_chunks = []

        for chunk, score in zip(chunks, scores):
            scored_chunk = dict(chunk)
            scored_chunk["rerank_score"] = float(score)
            scored_chunks.append(scored_chunk)

        # Step 4: sort chunks from highest score to lowest.
        scored_chunks.sort(
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )

        # Step 5: keep only the best final chunks.
        top_chunks = scored_chunks[:top_k]

        return top_chunks