# app/itineraries/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional


class TravelProfile(BaseModel):
    """
    Représente la demande utilisateur pour générer un itinéraire.
    city : nom de la ville principale (ex: "Marrakech"). Peut être None pour recherche multi-villes.
    duration_days : nombre de jours (int > 0).
    budget : "low" | "medium" | "high"
    interests : liste de centres d'intérêt (culture, gastronomy, shopping, nature, relax, nightlife, etc.)
    constraints : texte libre pour contrainte (avec enfants, peu de marche, etc.)
    language : code langue souhaitée ("fr", "en", ...)
    """
    city: Optional[str] = Field(default=None, description="Ville principale (optionnel).")
    duration_days: int = Field(..., gt=0, description="Durée du séjour en jours (>0).")
    budget: str = Field(..., description="Niveau de budget: low, medium, high")
    interests: List[str] = Field(default_factory=list, description="Centres d'intérêt")
    constraints: Optional[str] = Field(default=None, description="Contraintes (texte libre)")
    language: str = Field(default="fr", description="Langue du résultat (fr, en, ...)")

    @validator("budget")
    def budget_must_be_valid(cls, v):
        if v not in {"low", "medium", "high"}:
            raise ValueError("budget must be one of: 'low', 'medium', 'high'")
        return v


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
    # Optionnel : metadata source (id, city) si besoin ultérieurement
    source_id: Optional[str] = None
    source_city: Optional[str] = None


class ItineraryDay(BaseModel):
    day_number: int
    morning: List[ItineraryActivity] = Field(default_factory=list)
    afternoon: List[ItineraryActivity] = Field(default_factory=list)
    evening: List[ItineraryActivity] = Field(default_factory=list)


class Itinerary(BaseModel):
    city: Optional[str] = None
    duration_days: int
    profile: TravelProfile
    days: List[ItineraryDay] = Field(default_factory=list)
