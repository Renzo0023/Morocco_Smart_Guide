# app/data/loader.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import csv

from pydantic import BaseModel

from app.config import DATA_DIR

# LangChain: compatibilité ancienne / nouvelle version
try:
    from langchain_core.documents import Document
except ImportError:  # ancienne version
    from langchain.schema import Document


# -----------------------------
# Modèle de données : Place
# -----------------------------

class Place(BaseModel):
    """
    Représente un lieu / activité touristique au Maroc.

    Ce modèle est générique :
    - multi-ville (city varie : Marrakech, Fès, Essaouira, ...)
    - réutilisable pour les itinéraires, le RAG et l'API.
    """
    id: str
    name: str
    city: str
    country: str = "Morocco"

    category: Optional[str] = None        # ex: "culture", "nature", "shopping"
    description: Optional[str] = None
    budget: Optional[str] = None          # ex: "low", "medium", "high"
    duration_hours: Optional[float] = None
    best_time: Optional[str] = None       # ex: "morning", "afternoon", "evening"

    tags: List[str] = []                  # ex: ["culture", "famille", "gastronomie"]
    tips: Optional[str] = None            # conseils pratiques

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    extra: Dict[str, Any] = {}            # pour tous les champs supplémentaires


# -----------------------------
# Helpers pour CSV
# -----------------------------

def _detect_delimiter(path: Path) -> str:
    """
    Essaie de détecter automatiquement le délimiteur du CSV (: ',' ou ';').
    Si échec -> virgule par défaut.
    """
    with path.open("r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
            return dialect.delimiter
        except csv.Error:
            return ","


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise les clés (en minuscules, sans espaces).
    Exemple : 'Duration_Hours ' -> 'duration_hours'
    """
    normalized: Dict[str, Any] = {}
    for k, v in row.items():
        if k is None:
            continue
        key = k.strip().lower().replace(" ", "_")
        normalized[key] = v
    return normalized


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _parse_tags(value: Any) -> List[str]:
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    # On accepte ',' ou ';' comme séparateurs
    sep = "," if "," in s else ";"
    return [t.strip() for t in s.split(sep) if t.strip()]


def _load_places_from_csv(path: Path) -> List[Place]:
    """
    Charge un fichier CSV de lieux et le convertit en liste de Place.
    """
    #delimiter = _detect_delimiter(path)

    places: List[Place] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for raw_row in reader:
            row = _normalize_row(raw_row)
            print("Row normalisée:", row)

            id_ = row.get("id") or row.get("code") or row.get("slug") or row.get("name")
            if not id_:
                # Si vraiment rien, on skip la ligne
                continue

            name = row.get("name") or "Lieu sans nom"
            city = (row.get("city") or "Ville inconnue").strip()
            country = row.get("country") or "Morocco"

            category = row.get("category")
            description = row.get("description")
            budget = row.get("budget")
            duration_hours = _parse_float(row.get("duration_hours") or row.get("duration"))
            best_time = row.get("best_time")

            tags = _parse_tags(row.get("tags"))
            tips = row.get("tips")

            latitude = _parse_float(row.get("latitude"))
            longitude = _parse_float(row.get("longitude"))

            # Tout ce qui n'est pas dans la liste des champs connus va dans 'extra'
            used_keys = {
                "id",
                "code",
                "slug",
                "name",
                "city",
                "country",
                "category",
                "description",
                "budget",
                "duration_hours",
                "duration",
                "best_time",
                "tags",
                "tips",
                "latitude",
                "longitude",
            }

            extra = {
                k: v
                for k, v in row.items()
                if k not in used_keys
            }

            place = Place(
                id=str(id_),
                name=name,
                city=city,
                country=country,
                category=category,
                description=description,
                budget=budget,
                duration_hours=duration_hours,
                best_time=best_time,
                tags=tags,
                tips=tips,
                latitude=latitude,
                longitude=longitude,
                extra=extra,
            )
            places.append(place)

    return places


# -----------------------------
# API publique : chargement multi-villes
# -----------------------------

def load_places(data_dir: Path | None = None) -> List[Place]:
    """
    Charge tous les lieux à partir de tous les fichiers CSV dans un dossier.

    Convention :
      - Tous les fichiers du type *_places.csv dans DATA_DIR sont chargés.
      - Exemple :
          data/marrakech_places.csv
          data/fes_places.csv
          data/essaouira_places.csv
          data/chefchaouen_places.csv
    """
    base_dir = data_dir or DATA_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Dossier de données introuvable : {base_dir}")

    csv_files = sorted(base_dir.glob("*_places.csv"))
    print("CSV détectés:", csv_files)
    if not csv_files:
        # fallback : charger n'importe quel .csv si *_places.csv n'existe pas
        csv_files = sorted(base_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans {base_dir}. "
            f"Attendu : fichiers comme 'marrakech_places.csv', 'fes_places.csv', etc."
        )

    all_places: List[Place] = []
    for path in csv_files:
        subset = _load_places_from_csv(path)
        all_places.extend(subset)

    return all_places


# -----------------------------
# Conversion en Documents (RAG)
# -----------------------------

def place_to_document(place: Place) -> Document:
    """
    Convertit un Place en Document pour le vector store (FAISS).

    - page_content : le texte que le modèle va "lire" pour les embeddings.
    - metadata : infos structurées pour filtrer / afficher les résultats.
    """
    lines = [
        f"Nom: {place.name}",
        f"Ville: {place.city} ({place.country})",
    ]
    if place.category:
        lines.append(f"Catégorie: {place.category}")
    if place.budget:
        lines.append(f"Budget: {place.budget}")
    if place.duration_hours is not None:
        lines.append(f"Durée recommandée (heures): {place.duration_hours}")
    if place.best_time:
        lines.append(f"Meilleur moment: {place.best_time}")
    if place.description:
        lines.append(f"Description: {place.description}")
    if place.tips:
        lines.append(f"Conseils: {place.tips}")
    if place.tags:
        lines.append(f"Tags: {', '.join(place.tags)}")

    page_content = "\n".join(lines)

    metadata: Dict[str, Any] = {
        "id": place.id,
        "name": place.name,
        "city": place.city,
        "country": place.country,
        "category": place.category,
        "budget": place.budget,
        "duration_hours": place.duration_hours,
        "best_time": place.best_time,
        "tags": place.tags,
        "latitude": place.latitude,
        "longitude": place.longitude,
    }
    # On ajoute aussi extra dans la metadata pour ne rien perdre
    metadata.update({f"extra_{k}": v for k, v in place.extra.items()})

    return Document(page_content=page_content, metadata=metadata)


def to_documents(places: List[Place]) -> List[Document]:
    """
    Transforme une liste de Place en liste de Documents pour le RAG.
    """
    return [place_to_document(p) for p in places]

