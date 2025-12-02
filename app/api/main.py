# app/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
#  🔧 SESSIONS AVEC MÉMOIRE PERSISTANTE
# =====================================================

chat_sessions = {}   # { session_id: { chain, memory } }


def create_new_session() -> str:
    session_id = str(uuid.uuid4())
    chain = get_rag_conversation_chain()

    chat_sessions[session_id] = {
        "chain": chain,
        "history": []
    }
    return session_id


def get_session(session_id: str):
    session = chat_sessions.get(session_id)
    if session is None:
        session_id = create_new_session()
        session = chat_sessions[session_id]
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
    try:
        return generate_itinerary(profile)
    except Exception as e:
        raise HTTPException(500, f"Erreur génération itinéraire : {e}")


# -----------------------------------------------------
#  💬 CHATBOT AVEC MEMOIRE
# -----------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat_with_assistant(request: ChatRequest):

    if request.session_id is None:
        session_id = create_new_session()
    else:
        session_id, session = get_session(request.session_id)

    session = chat_sessions[session_id]
    chain = session["chain"]

    try:
        result = chain({"question": request.message})
        answer = result["answer"]
    except Exception as e:
        raise HTTPException(500, f"Erreur interne chat : {e}")

    session["history"].append({
        "user": request.message,
        "assistant": answer
    })

    return ChatResponse(
        session_id=session_id,
        answer=answer
    )


# -----------------------------------------------------
#  ⭐ NOUVEAU : RECOMMANDATIONS AVANT ITINERAIRE
# -----------------------------------------------------
@app.get("/recommendations")
def get_recommendations(city: str, interests: str = "", k: int = 10):
    """
    Recommander des lieux en fonction :
    - d'une ville
    - d'intérêts (culture, nature, gastronomy, shopping...)
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

