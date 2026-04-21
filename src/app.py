import os
import sys
import tempfile
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from ingest import ingest_file, load_vectorstore, ingest_from_wikipedia, ingest_from_url
from rag_chain import ask

st.set_page_config(
    page_title="Medical RAG",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

with st.sidebar:
    st.title("Medical RAG")
    st.caption("Ollama + ChromaDB + Streamlit")
    st.divider()

    st.subheader("Settings")

    model_name = st.selectbox(
        "Ollama model",
        ["llama3", "mistral", "phi3"],
    )

    chunk_size = st.slider("Chunk size", 200, 1500, 500, 50)
    top_k = st.slider("Top-k chunks", 1, 8, 3)

    st.divider()

    tab1, tab2 = st.tabs(["Upload PDF", "Web Source"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload medical PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            new_files = [
                f for f in uploaded_files
                if f.name not in st.session_state.ingested_files
            ]
            if new_files:
                with st.spinner(f"Indexing {len(new_files)} PDF(s)..."):
                    for uploaded_file in new_files:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        vs, n_chunks = ingest_file(tmp_path, chunk_size=chunk_size)
                        st.session_state.vectorstore = vs
                        st.session_state.ingested_files.append(uploaded_file.name)
                        os.unlink(tmp_path)
                st.success(f"Indexed {len(new_files)} PDF(s)!")

    with tab2:
        st.markdown("**Wikipedia**")
        wiki_query = st.text_input(
            "Search term",
            placeholder="e.g. ischemic stroke",
            key="wiki_query",
        )
        if st.button("Fetch from Wikipedia", use_container_width=True):
            if wiki_query.strip() == "":
                st.warning("Enter a search term first.")
            else:
                with st.spinner(f"Fetching '{wiki_query}' from Wikipedia..."):
                    try:
                        vs, n_chunks, titles = ingest_from_wikipedia(wiki_query)
                        st.session_state.vectorstore = vs
                        for title in titles:
                            if title not in st.session_state.ingested_files:
                                st.session_state.ingested_files.append(title)
                        st.success(f"Indexed {n_chunks} chunks from Wikipedia!")
                    except Exception as e:
                        st.error(f"Wikipedia fetch failed: {e}")

        st.divider()

        st.markdown("**Any webpage URL**")
        st.caption("PubMed abstract, NHS page, medical journal...")
        web_url = st.text_input(
            "Paste URL",
            placeholder="https://pubmed.ncbi.nlm.nih.gov/...",
            key="web_url",
        )
        if st.button("Fetch from URL", use_container_width=True):
            if web_url.strip() == "":
                st.warning("Paste a URL first.")
            elif not web_url.startswith("http"):
                st.warning("URL must start with http:// or https://")
            else:
                with st.spinner("Fetching page content..."):
                    try:
                        vs, n_chunks = ingest_from_url(web_url)
                        st.session_state.vectorstore = vs
                        if web_url not in st.session_state.ingested_files:
                            st.session_state.ingested_files.append(web_url)
                        st.success(f"Indexed {n_chunks} chunks from URL!")
                    except Exception as e:
                        st.error(f"Could not fetch URL: {e}")

    if st.session_state.ingested_files:
        st.divider()
        st.caption("Indexed sources:")
        for fname in st.session_state.ingested_files:
            st.markdown(f"- `{fname}`")

    if st.session_state.vectorstore is None and os.path.exists("vectorstore"):
        st.session_state.vectorstore = load_vectorstore()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.header("Medical Document Q&A")

if not st.session_state.ingested_files and not os.path.exists("vectorstore"):
    st.info("Get started by uploading a PDF or fetching from Wikipedia/URL in the sidebar.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{os.path.basename(src['source'])}** · page {src['page']}")
                    st.caption(src["snippet"])

if prompt := st.chat_input("Ask about your medical documents..."):
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF or fetch a web source first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = ask(
                question=prompt,
                vectorstore=st.session_state.vectorstore,
                model_name=model_name,
                k=top_k,
            )
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources", expanded=True):
                for src in result["sources"]:
                    st.markdown(f"**{os.path.basename(src['source'])}** · page {src['page']}")
                    st.caption(src["snippet"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
