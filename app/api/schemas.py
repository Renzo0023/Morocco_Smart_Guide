# app/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """
    Requête envoyée par le frontend pour discuter avec le chatbot.
    """
    session_id: Optional[str] = Field(
        default=None,
        description="Identifiant de session pour la conversation."
    )
    message: str = Field(..., description="Message envoyé par l'utilisateur.")


class ChatResponse(BaseModel):
    """
    Réponse du chatbot, incluant l'ID de session pour la mémoire.
    """
    session_id: str = Field(..., description="Identifiant de la session active.")
    answer: str = Field(..., description="Réponse générée par l'assistant.")
