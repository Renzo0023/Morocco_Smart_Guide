from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid

from app.api.schemas import ChatRequest, ChatResponse
from app.itineraries.models import TravelProfile, Itinerary
from app.itineraries.generator import generate_itinerary
from app.rag.qa_chain import get_rag_chain


# ===========================
# INITIALISATION FASTAPI
# ===========================

app = FastAPI(
    title="Morocco Smart Guide API",
    description="Backend IA pour itinéraires et chatbot multi-villes au Maroc",
    version="1.1.0",
)

# Autoriser les requêtes depuis le front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================
# GESTION DES SESSIONS DE CHAT
# ===========================

chat_sessions = {}  # { session_id: { rag_chain, history } }


def create_new_session() -> str:
    """
    Crée une nouvelle session de chat et retourne l'ID.
    """
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        "rag_chain": get_rag_chain(),
        "history": []
    }
    return session_id


def get_session(session_id: str):
    """
    Renvoie (session_id, data). Si inconnue -> nouvelle session.
    """
    session = chat_sessions.get(session_id)
    if session is None:
        session_id = create_new_session()
        session = chat_sessions[session_id]
    return session_id, session


# ===========================
# ENDPOINTS API
# ===========================

@app.get("/health")
def health_check():
    return {"status": "ok"}


# -------- ITINÉRAIRE -------- #

@app.post("/itinerary", response_model=Itinerary)
def create_itinerary(profile: TravelProfile):
    """
    Génère un itinéraire pour n'importe quelle ville du Maroc.
    """
    try:
        itinerary = generate_itinerary(profile)
        return itinerary
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de l'itinéraire : {e}"
        )


# -------- CHATBOT -------- #

@app.post("/chat", response_model=ChatResponse)
def chat_with_assistant(request: ChatRequest):
    """
    Chatbot IA basé RAG couvrant toutes les villes présentes dans le dataset.
    """

    # 1. Gestion session
    if request.session_id is None:
        session_id = create_new_session()
    else:
        session_id, session = get_session(request.session_id)

    session = chat_sessions[session_id]
    rag_chain = session["rag_chain"]

    # 2. Appel au modèle
    try:
        result = rag_chain.invoke({"query": request.message})
        answer = result["result"]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur dans le chat : {e}"
        )

    # 3. Mémoire locale
    session["history"].append({
        "user": request.message,
        "assistant": answer
    })

    # 4. Retour
    return ChatResponse(
        session_id=session_id,
        answer=answer
    )

