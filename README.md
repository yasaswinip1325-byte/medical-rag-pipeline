# Medical RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) pipeline for medical documents, built with LangChain, ChromaDB, Ollama, and Streamlit. No API keys required — everything runs on your machine.

## What it does

Upload clinical PDFs or fetch content from Wikipedia / any public URL (PubMed, NHS, medical journals) and ask questions in natural language. The system retrieves the most relevant passages and generates grounded answers with source citations — preventing hallucination by anchoring the LLM strictly to your documents.

A medical keyword guard rejects non-clinical questions before they reach the LLM.

## Architecture

```
PDF / Wikipedia / URL
        ↓
   Chunking (RecursiveCharacterTextSplitter)
        ↓
   Embedding (nomic-embed-text via Ollama)
        ↓
   ChromaDB (local vector store)
        ↓
User query → embed → cosine similarity search → top-k chunks
        ↓
   Medical keyword guard (reject non-medical questions)
        ↓
   Prompt builder (query + labelled context)
        ↓
   Ollama LLM (llama3 / mistral — fully local)
        ↓
   Answer + source citations
```

## Tech stack

| Component | Tool |
|---|---|
| Document loading | LangChain PyPDFLoader, WikipediaLoader, WebBaseLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | nomic-embed-text via Ollama |
| Vector store | ChromaDB (local, file-based) |
| LLM | Ollama — llama3, mistral, phi3 |
| UI | Streamlit |

## Setup

**1. Install Ollama** from https://ollama.com, then pull models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

**2. Clone and install:**

```bash
git clone https://github.com/<your-username>/medical-rag-pipeline.git
cd medical-rag-pipeline
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install wikipedia
```

**3. Run:**

```bash
streamlit run src/app.py
```

Open http://localhost:8501

## Usage

**Option A — Upload PDF:**
Upload any medical PDF (clinical guidelines, research papers, drug references) via the Upload PDF tab in the sidebar.

**Option B — Web source:**
- Type a search term (e.g. `ischemic stroke`) to fetch from Wikipedia
- Or paste any public URL (PubMed abstract, NHS page, medical journal)

Then ask questions in the chat. Expand "Sources" under each answer to see exactly which document pages were used.

## Key concepts

**Chunking** — Documents split into overlapping 500-char passages so retrieval is precise and no sentence is cut at a boundary.

**Embeddings** — Each chunk converted to a 768-dimensional vector using nomic-embed-text. Semantically similar text produces similar vectors.

**Cosine similarity** — Query embedded and compared against all stored chunk vectors. Closest k chunks retrieved.

**Prompt engineering** — Retrieved chunks injected into a structured prompt instructing the LLM to answer ONLY from provided context, eliminating hallucination.

**Medical keyword guard** — Non-medical questions rejected before hitting the LLM, keeping the assistant focused on clinical queries.

## Project structure

```
medical-rag-pipeline/
├── README.md
├── requirements.txt
├── data/                 ← put sample PDFs here for testing
├── src/
│   ├── ingest.py         ← PDF / Wikipedia / URL loading, chunking, embedding
│   ├── retriever.py      ← cosine similarity search + context formatting
│   ├── rag_chain.py      ← medical guard + prompt template + Ollama LLM
│   └── app.py            ← Streamlit chat UI with dual ingestion tabs
└── vectorstore/          ← ChromaDB persists here (auto-created)
```

## Author

Parsa Yasaswini — AI/ML Engineer
[LinkedIn](https://linkedin.com/in/parsa-yasaswini65) · yasaswini.p1325@gmail.com
