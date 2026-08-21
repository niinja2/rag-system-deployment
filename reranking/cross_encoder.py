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
        top_k: int = 5) -> list[dict]:

        pairs = []

        for chunk in chunks:
            chunk_text = chunk["text"]
            pair = (query, chunk_text)
            pairs.append(pair)

        scores = self.model.predict(pairs)

        scored_chunks = []

        for chunk, score in zip(chunks, scores):
            scored_chunk = dict(chunk)
            scored_chunk["rerank_score"] = float(score)
            scored_chunks.append(scored_chunk)

        scored_chunks.sort(
            key=lambda chunk: chunk["rerank_score"],
            reverse=True,
        )

        top_chunks = scored_chunks[:top_k]

        return top_chunks