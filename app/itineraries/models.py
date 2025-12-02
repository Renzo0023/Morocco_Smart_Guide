# app/itineraries/models.py

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class TravelProfile(BaseModel):
    """
    Représente la demande utilisateur pour générer un itinéraire.

    - city : nom de la ville principale (ex: "Marrakech"). Peut être None pour une recherche multi-villes.
    - duration_days : nombre de jours (int > 0).
    - budget : "low" | "medium" | "high".
    - interests : liste de centres d'intérêt (culture, gastronomy, shopping, nature, relax, nightlife, etc.).
    - constraints : texte libre pour contrainte (avec enfants, peu de marche, etc.).
    - language : code langue souhaitée ("fr", "en", ...).
    """
    city: Optional[str] = Field(default=None, description="Ville principale (optionnelle).")
    duration_days: int = Field(..., gt=0, description="Durée du séjour en jours (>0).")
    budget: str = Field(..., description="Niveau de budget: low, medium, high.")
    interests: List[str] = Field(default_factory=list, description="Centres d'intérêt.")
    constraints: Optional[str] = Field(default=None, description="Contraintes (texte libre).")
    language: str = Field(default="fr", description="Langue du résultat (fr, en, ...).")

    @validator("budget")
    def budget_must_be_valid(cls, v: str) -> str:
        allowed = {"low", "medium", "high"}
        if v not in allowed:
            raise ValueError(f"budget must be one of: {', '.join(sorted(allowed))}")
        return v

    @validator("interests", pre=True)
    def normalize_interests(cls, v):
        # Permet d'accepter une string "culture, gastronomy" ou une vraie liste
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []


class ItineraryActivity(BaseModel):
    """
    Activité / lieu dans l'itinéraire.
    Les champs correspondent à ce que le LLM doit renvoyer dans le JSON.
    """
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    budget: Optional[str] = None
    best_time: Optional[str] = None
    tips: Optional[str] = None

    # Optionnel : metadata source (id, city) pour rattacher l'activité à la base RAG
    source_id: Optional[str] = None
    source_city: Optional[str] = None


class ItineraryDay(BaseModel):
    """
    Représente une journée de l'itinéraire.
    """
    day_number: int
    morning: List[ItineraryActivity] = Field(default_factory=list)
    afternoon: List[ItineraryActivity] = Field(default_factory=list)
    evening: List[ItineraryActivity] = Field(default_factory=list)


class Itinerary(BaseModel):
    """
    Itinéraire complet généré pour un profil donné.
    """
    city: Optional[str] = None
    duration_days: int
    profile: TravelProfile
    days: List[ItineraryDay] = Field(default_factory=list)
    notes: Optional[str] = None  # remarques globales éventuelles (optionnel, rempli par le LLM)
