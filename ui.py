import streamlit as st
import requests

st.title("Production RAG System")

query = st.text_input("Ask a question", placeholder="e.g. What is the capital of France?")

if st.button("Submit") and query:
    resp = requests.post(
        "http://localhost:8000/query",
        json={
            "query": query,
            "llm_provider": "openrouter",
            "use_reranker": True,
        },
    )
    data = resp.json()

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Sources")
    for chunk_id, text in zip(data["sources"], data["contexts"]):
        with st.expander(f"Chunk {chunk_id}"):
            st.write(text)
