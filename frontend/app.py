import os

import requests
import streamlit as st

API_URL = os.getenv("RESEARCHPILOT_API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="ResearchPilot", page_icon="📚", layout="wide")
st.title("📚 ResearchPilot")
st.caption("Research assistant with BM25, dense/hybrid retrieval, LangGraph and cited answers")
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload research PDF", type=["pdf"])
    if uploaded and st.button("Index document", use_container_width=True):
        try:
            with st.spinner("Indexing document..."):
                res = requests.post(
                    f"{API_URL}/documents/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=300,
                )
                res.raise_for_status()
                st.success(f"Indexed {res.json()['filename']}")
        except requests.RequestException as exc:
            st.error(str(exc))
    mode = st.selectbox("Retrieval method", ["hybrid", "dense", "bm25"])
    attempts = st.slider("Max retrieval attempts", 1, 5, 3)
    try:
        res = requests.get(f"{API_URL}/documents", timeout=20)
        res.raise_for_status()
        for d in res.json():
            st.write("📄", d["filename"])
    except requests.RequestException:
        st.warning("Backend unavailable")


def render(message):
    st.markdown(message["content"])
    if message.get("sources"):
        with st.expander("Source citations", expanded=True):
            for source in message["sources"]:
                st.markdown(f"**[{source['id']}] {source['document']} — page {source['page']}**")
                st.write(source["text"])
    if message.get("trace"):
        with st.expander("Agent action trace"):
            for event in message["trace"]:
                st.text(event)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render(message)

question = st.chat_input("Ask a question about your PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and grading evidence..."):
                res = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "retrieval_mode": mode, "max_attempts": attempts},
                    timeout=300,
                )
                res.raise_for_status()
                data = res.json()
            message = {
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"],
                "trace": data["trace"],
            }
            render(message)
            st.session_state.messages.append(message)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
