import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from retriever import retrieve_chunks, format_context

MEDICAL_KEYWORDS = [
    "symptom", "treatment", "diagnosis", "disease", "drug",
    "patient", "clinical", "medical", "dose", "therapy",
    "stroke", "cancer", "infection", "surgery", "pain",
    "blood", "heart", "brain", "lung", "kidney", "MRI",
    "scan", "imaging", "medicine", "condition", "disorder",
    "chronic", "acute", "onset", "prognosis", "pathology",
    "anatomy", "physiology", "neurology", "cardiology",
    "radiology", "oncology", "pediatric", "psychiatric",
    "pharmaceutical", "antibiotic", "vaccine", "virus",
    "bacteria", "inflammation", "injury", "fracture", "wound"
]


def is_medical_question(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in MEDICAL_KEYWORDS)


MEDICAL_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a clinical AI assistant. Answer the question below
using ONLY the information provided in the context. Be precise and cite
which source your answer comes from.

If the context does not contain enough information to answer the question,
say: "I couldn't find this in the uploaded documents."

Do NOT use prior medical knowledge outside of the context provided.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)


def build_rag_chain(model_name: str = "llama3"):
    llm = OllamaLLM(model=model_name, temperature=0.1)
    chain = MEDICAL_RAG_PROMPT | llm
    return chain


def ask(
    question: str,
    vectorstore: Chroma,
    model_name: str = "llama3",
    k: int = 3,
) -> dict:
    if not is_medical_question(question):
        return {
            "answer": (
                "This assistant is designed for medical document queries only. "
                "Please ask a medical or clinical question — for example about "
                "symptoms, treatments, diagnoses, or medications."
            ),
            "sources": [],
            "blocked": True,
        }

    docs = retrieve_chunks(question, vectorstore, k=k)
    context = format_context(docs)

    chain = build_rag_chain(model_name)
    answer = chain.invoke({"context": context, "question": question})

    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "snippet": doc.page_content[:120] + "...",
        })

    return {"answer": answer, "sources": sources, "blocked": False}
