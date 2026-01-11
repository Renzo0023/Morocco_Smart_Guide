from typing import List, Optional
from pydantic import BaseModel, Field, validator


class TravelProfile(BaseModel):
    """
    Représente la demande utilisateur pour générer un itinéraire.
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
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []

    class Config:
        extra = "ignore"   # <<< AJOUT IMPORTANT


class ItineraryActivity(BaseModel):
    """
    Activité / lieu dans l'itinéraire.
    """
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    budget: Optional[str] = None
    best_time: Optional[str] = None
    tips: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    # Metadata optionnelle
    source_id: Optional[str] = None
    source_city: Optional[str] = None

    # Ajout du lien Google Maps
    maps_url: Optional[str] = None

    class Config:
        extra = "ignore"   # <<< AJOUT IMPORTANT


class ItineraryDay(BaseModel):
    """
    Représente une journée de l'itinéraire.
    """
    day_number: int
    morning: List[ItineraryActivity] = Field(default_factory=list)
    afternoon: List[ItineraryActivity] = Field(default_factory=list)
    evening: List[ItineraryActivity] = Field(default_factory=list)

    class Config:
        extra = "ignore"   # <<< AJOUT IMPORTANT


class Itinerary(BaseModel):
    """
    Itinéraire complet généré pour un profil donné.
    """
    city: Optional[str] = None
    duration_days: int
    profile: TravelProfile
    days: List[ItineraryDay] = Field(default_factory=list)
    notes: Optional[str] = None

    class Config:
        extra = "ignore"   # <<< AJOUT IMPORTANT
