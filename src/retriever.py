from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def get_retriever(vectorstore: Chroma, k: int = 3):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_chunks(query: str, vectorstore: Chroma, k: int = 3):
    retriever = get_retriever(vectorstore, k)
    docs = retriever.invoke(query)

    print(f"Query: '{query[:60]}'")
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"  [{i+1}] {src} page {page}")

    return docs


def format_context(docs) -> str:
    context_parts = []
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        label = f"[Source {i+1}: {src}, page {page}]"
        context_parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)
