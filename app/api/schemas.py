from pydantic import BaseModel, Field
from typing import Optional

from app.itineraries.models import TravelProfile, Itinerary


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Identifiant de session pour la conversation."
    )
    message: str = Field(..., description="Message envoyé par l'utilisateur.")
    language: str = Field(
        default="fr",
        description="Langue de la réponse souhaitée (fr/en)."
    )


class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Identifiant de la session active.")
    answer: str = Field(..., description="Réponse générée par l'assistant.")

