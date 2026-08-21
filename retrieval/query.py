import json
import faiss
import numpy as np
import config   
from sentence_transformers import SentenceTransformer



class Retriever:

    def __init__(self):
        self.index = self.load_index()
        self.metas = self.load_metadata()
        self.model = SentenceTransformer(config.MODEL_NAME)

    def load_index(self):
        return faiss.read_index(config.INDEX_PATH)

    def load_metadata(self):
        metas = []
        with open(config.META_PATH,"r", encoding="utf-8") as f:
            for line in f:
                metas.append(json.loads(line))
        return metas

    def embed_query(self, query):
        vec = self.model.encode([query])
        vec = np.array(vec, dtype="float32")
        faiss.normalize_L2(vec)
        return vec

    def search(self, query, top_k=5):

        q_vec = self.embed_query(query)
        scores, ids = self.index.search(q_vec, top_k)

        results = []

        for score, idx in zip(scores[0], ids[0]):

            meta = self.metas[idx]

            if "chunk_id" in meta and meta["chunk_id"]:
                chunk_id = meta["chunk_id"]

            elif "doc_id" in meta and meta["doc_id"]:
                chunk_id = meta["doc_id"]

            else:
                raise ValueError(
                    "Metadata must contain a non-empty chunk_id or doc_id."
                )
            
            d_dict = {
                 "chunk_id": chunk_id,
                 "chunk_id": meta["doc_id"],
                "text": meta["text"],
                "source_path": meta.get("source_path"),
                "chunk_index": meta.get("chunk_index"),
                "score": float(score),
            }

            results.append(d_dict)

        return results

