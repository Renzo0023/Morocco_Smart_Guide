# app/api/main.py

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import ChatRequest, ChatResponse
from app.itineraries.models import TravelProfile, Itinerary
from app.itineraries.generator import generate_itinerary
from app.rag.qa_chain import get_rag_conversation_chain
from app.rag.vectorstore import get_retriever


app = FastAPI(
    title="Morocco Smart Guide API",
    description="Backend IA pour itinéraires & chatbot multi-villes",
    version="1.3.0",
)

# CORS (pour que Streamlit/Gradio en front puissent appeler l'API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # pour un projet étudiant c'est ok
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
#  🔧 SESSIONS AVEC MÉMOIRE EN RAM (simple mais suffisant)
# =====================================================

# Dictionnaire { session_id: { "chain": ConversationalRetrievalChain, "history": [...] } }
chat_sessions: dict[str, dict] = {}


def create_new_session() -> str:
    """
    Crée une nouvelle session de chat :
    - nouvelle chaîne RAG + mémoire (via get_rag_conversation_chain)
    - historique vide
    """
    session_id = str(uuid.uuid4())
    chain = get_rag_conversation_chain()

    chat_sessions[session_id] = {
        "chain": chain,
        "history": [],
    }
    return session_id


def get_session(session_id: str) -> tuple[str, dict]:
    """
    Retourne (session_id_effectif, session_data).
    Si la session demandée n'existe pas, on crée une nouvelle session.
    """
    session = chat_sessions.get(session_id)
    if session is None:
        new_id = create_new_session()
        return new_id, chat_sessions[new_id]
    return session_id, session


# =====================================================
#  ENDPOINTS API
# =====================================================

@app.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------------------------------
#  🧭 GENERATION D’ITINERAIRE
# -----------------------------------------------------
@app.post("/itinerary", response_model=Itinerary)
def create_itinerary(profile: TravelProfile):
    """
    Génère un itinéraire complet à partir d'un TravelProfile.
    """
    try:
        itinerary = generate_itinerary(profile)
        return itinerary
    except Exception as e:
        raise HTTPException(500, f"Erreur génération itinéraire : {e}")


# -----------------------------------------------------
#  💬 CHATBOT AVEC MEMOIRE RAG
# -----------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_with_assistant(request: ChatRequest):
    """
    Chatbot touristique :
    - si pas de session_id -> création d'une nouvelle session avec mémoire
    - sinon on récupère la chaîne RAG associée
    """
    # Gestion de la session
    if request.session_id is None:
        session_id = create_new_session()
    else:
        session_id, _ = get_session(request.session_id)

    session = chat_sessions[session_id]
    chain = session["chain"]

    # Appel du RAG
    try:
        result = chain({"question": request.message})
        answer = result["answer"]
    except Exception as e:
        raise HTTPException(500, f"Erreur interne chat : {e}")

    # Mise à jour historique (au cas où tu veuilles l'afficher côté front)
    session["history"].append(
        {
            "user": request.message,
            "assistant": answer,
        }
    )

    return ChatResponse(
        session_id=session_id,
        answer=answer,
    )


# -----------------------------------------------------
#  ⭐ RECOMMANDATIONS DE LIEUX AVANT ITINERAIRE
# -----------------------------------------------------
@app.get("/recommendations")
def get_recommendations(city: str, interests: str = "", k: int = 10):
    """
    Recommander des lieux en fonction :
    - d'une ville
    - d'intérêts (culture, nature, gastronomy, shopping...)

    Utilise directement le retriever FAISS (RAG) sans LLM.
    """
    retriever = get_retriever(k=k, city=city)

    query = f"Lieux recommandés pour : {interests} à {city}"
    docs = retriever.get_relevant_documents(query)

    return [
        {
            "name": d.metadata.get("name"),
            "city": d.metadata.get("city"),
            "category": d.metadata.get("category"),
            "budget": d.metadata.get("budget"),
            "best_time": d.metadata.get("best_time"),
            "description": d.page_content,
        }
        for d in docs
    ]
