from pydantic import BaseModel, Field
from typing import Optional

from app.itineraries.models import TravelProfile, Itinerary


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Identifiant de session de chat. Si None, une nouvelle session sera créée."
    )
    message: str = Field(..., description="Message envoyé par l'utilisateur.")
    language: str = Field(
        default="fr",
        description="Langue souhaitée pour la réponse (fr, en, ...)."
    )


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="ID de session de chat utilisé")
    answer: str = Field(..., description="Réponse de l'assistant IA")

