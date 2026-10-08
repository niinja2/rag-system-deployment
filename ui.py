import os
import time

import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_resource
def wait_for_api():
    for _ in range(60):
        try:
            requests.get(f"{API_URL}/health", timeout=1)
            return True
        except requests.RequestException:
            time.sleep(2)
    return False


st.title("Production RAG System")

with st.spinner("API loading..."):
    if not wait_for_api():
        st.error("API not reachable. Check that the API is running and the data is downloaded, then refresh.")
        st.stop()

llm_provider = st.selectbox("LLM Provider", ["mock", "openrouter", "ollama"])
query = st.text_input("Ask a question", placeholder="e.g. What is the capital of France?")

if st.button("Submit") and query:
    resp = requests.post(
        f"{API_URL}/query",
        json={
            "query": query,
            "llm_provider": llm_provider,
            "use_reranker": True,
        },
    )
    if not resp.ok:
        st.error(
            f"Request failed (HTTP {resp.status_code}). "
            "Check that the selected provider is available."
        )
        st.stop()

    data = resp.json()

    st.caption(f"Model: {data['model']}  |  Server: {data['server']}")

    st.subheader("Answer")
    st.write(data["answer"])

    st.subheader("Sources")
    for chunk_id, text in zip(data["sources"], data["contexts"]):
        with st.expander(f"Chunk {chunk_id}"):
            st.write(text)
