# app/rag/qa_chain.py

from __future__ import annotations

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# LLM Hugging Face (hébergé)
try:
    from langchain_community.llms import HuggingFaceHub
except ImportError:
    from langchain.llms import HuggingFaceHub  # fallback anciennes versions

from app.config import HF_API_KEY, LLM_MODEL_NAME
from app.rag.vectorstore import get_retriever


# -----------------------------
# LLM Hugging Face
# -----------------------------

def get_llm():
    """
    Retourne un LLM Hugging Face via l'API Inference.

    - Modèle défini dans LLM_MODEL_NAME (config.py), ex :
        "mistralai/Mistral-7B-Instruct-v0.2"
    - Nécessite un token HF_API_KEY dans .env

    Si tu veux passer à un modèle local plus tard,
    tu pourras remplacer cette fonction par un HuggingFacePipeline local.
    """
    if not HF_API_KEY:
        raise ValueError(
            "HF_API_KEY manquant. Ajoute-le dans ton fichier .env pour utiliser HuggingFaceHub."
        )

    llm = HuggingFaceHub(
        repo_id=LLM_MODEL_NAME,
        huggingfacehub_api_token=HF_API_KEY,
        model_kwargs={
            "temperature": 0.4,
            "max_new_tokens": 512,
        },
    )
    return llm


# -----------------------------
# Chaîne RAG conversationnelle
# -----------------------------

def get_rag_conversation_chain(k: int = 5):
    """
    Crée une chaîne RAG conversationnelle avec :

    - Retriever FAISS (multi-villes)
    - LLM Hugging Face
    - Mémoire de conversation (historique)
    - Prompt spécialisé tourisme au Maroc
    """
    retriever = get_retriever(k=k)
    llm = get_llm()

    # Mémoire : garde tout l'historique de la session
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    # Prompt personnalisé pour combiner contexte + question
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Tu es un expert du tourisme au Maroc (toutes villes : Marrakech, Fès, Tanger, "
            "Agadir, Rabat, Chefchaouen, Essaouira, etc.).\n"
            "Réponds uniquement en utilisant le contexte fourni.\n\n"
            "=== CONTEXTE ===\n{context}\n"
            "=== QUESTION ===\n{question}\n\n"
            "Réponds dans un style clair, humain et utile. "
            "Si l'information n'est pas dans le contexte, dis-le explicitement."
        ),
    )

    # Chaîne RAG + mémoire
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return chain
