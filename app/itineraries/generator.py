# app/itineraries/generator.py

import json
from typing import List, Optional

# Compatibilité Document selon la version de LangChain
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from app.itineraries.models import (
    TravelProfile,
    Itinerary,
    ItineraryDay,
    ItineraryActivity,
)

# NOTE: On importe get_retriever depuis le vectorstore (comme prévu dans ton P2).
try:
    from app.rag.vectorstore import get_retriever  # recommandé
except Exception:
    # Si un jour tu déplaces get_retriever dans un autre module, adapter ici.
    raise ImportError(
        "Missing get_retriever() dans app.rag.vectorstore. "
        "P2 doit exposer une fonction get_retriever() qui renvoie un retriever LangChain."
    )

# On réutilise le LLM Hugging Face défini dans qa_chain.py
from app.rag.qa_chain import get_llm


# ---------- Helpers RAG / sélection de lieux ----------

def build_retriever_query(profile: TravelProfile) -> str:
    """
    Construit une requête textuelle pour interroger FAISS selon le profil voyageur.
    """
    interests = ", ".join(profile.interests) if profile.interests else "général"
    constraints = profile.constraints or "aucune contrainte particulière"
    city_clause = f"Ville: {profile.city}." if getattr(profile, "city", None) else "Multi-ville."

    return (
        f"{city_clause} Séjour de {profile.duration_days} jours. "
        f"Budget: {profile.budget}. Centres d'intérêt: {interests}. "
        f"Contraintes: {constraints}."
    )


def get_candidate_places(profile: TravelProfile, max_docs: int = 30) -> List[Document]:
    """
    Utilise le retriever FAISS pour obtenir des documents candidats.

    Si les documents ont la metadata 'city' et que profile.city est renseigné,
    on filtre pour ne garder que ceux de la ville souhaitée.
    """
    retriever = get_retriever()
    query = build_retriever_query(profile)

    # Utiliser la méthode standard du retriever pour obtenir les docs
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
    elif hasattr(retriever, "get_relevant_documents_by_query"):
        docs = retriever.get_relevant_documents_by_query(query)
    else:
        # fallback : similarity_search
        docs = retriever.similarity_search(query, k=max_docs if max_docs else 10)

    # Filtrer par city metadata si demandé
    if getattr(profile, "city", None):
        filtered: List[Document] = []
        for d in docs:
            meta_city = None
            if isinstance(d.metadata, dict):
                meta_city = (
                    d.metadata.get("city")
                    or d.metadata.get("location")
                    or d.metadata.get("source_city")
                )
                if meta_city and isinstance(meta_city, str):
                    if profile.city.lower() in meta_city.lower():
                        filtered.append(d)
                else:
                    # si pas de ville en metadata, on conserve (comportement conservatif)
                    filtered.append(d)
            else:
                filtered.append(d)
        docs = filtered

    return docs[:max_docs]


def format_places_for_prompt(docs: List[Document], max_chars: int = 20000) -> str:
    """
    Transforme les documents en un bloc de texte lisible par le LLM.
    Tronque si trop long (max_chars).
    """
    parts = []
    for d in docs:
        meta = d.metadata or {}
        name = meta.get("name") or meta.get("title") or (d.page_content[:50] + "...")
        city = meta.get("city") or meta.get("location") or ""
        category = meta.get("category") or ""
        budget = meta.get("budget") or ""
        best_time = meta.get("best_time") or ""
        parts.append(
            f"- {name} | city: {city} | category: {category} | budget: {budget} | best_time: {best_time}\n"
            f"  Description: {d.page_content}\n"
        )
    text = "\n".join(parts)
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def build_itinerary_prompt(profile: TravelProfile, places_text: str) -> str:
    """
    Construit un prompt explicite demandant une sortie JSON strictement formatée.
    """
    interests = ", ".join(profile.interests) if profile.interests else "général"
    constraints = profile.constraints or "aucune contrainte particulière"

    prompt = f"""
Tu es un expert en tourisme au Maroc et en planification de voyages.

Contrainte utilisateur:
- Ville principale: {profile.city or 'Multi-villes (pas de ville spécifiée)'}
- Durée (jours): {profile.duration_days}
- Budget: {profile.budget}
- Centres d'intérêt: {interests}
- Contraintes: {constraints}
- Langue souhaitée: {profile.language}

Voici la liste des lieux / activités disponibles (utilise uniquement ces éléments pour les recommandations):
{places_text}

Instructions:
1) Conçois un itinéraire pour {profile.duration_days} jours.
2) Chaque jour doit contenir au minimum une activité le matin et une l'après-midi; le soir est optionnel.
3) Respecte le budget et les centres d'intérêt autant que possible.
4) N'ajoute pas de lieux qui ne figurent pas dans la liste ci-dessus. Tu peux regrouper plusieurs activités proches le même jour.
5) Réponds STRICTEMENT en JSON valide avec la structure suivante:

{{
  "city": "{profile.city or ''}",
  "days": [
    {{
      "day_number": 1,
      "morning": [
        {{
          "name": "...",
          "category": "...",
          "description": "...",
          "budget": "...",
          "best_time": "...",
          "tips": "...",
          "source_id": "...",
          "source_city": "..."
        }}
      ],
      "afternoon": [...],
      "evening": [...]
    }}
  ],
  "notes": "optionnel: remarques globales sur l'itinéraire"
}}

IMPORTANT:
- Ne fournis aucun texte hors du JSON.
- Si tu dois mentionner un manque d'information, place le message dans le champ "notes" au niveau du JSON.
"""
    return prompt


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Tente d'extraire le premier objet JSON complet trouvé dans `text`.
    Renvoie la string JSON ou None.

    Méthode robuste: cherche la première '{' et balaye jusqu'à équilibrer toutes les accolades.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_itinerary_json(json_text: str, profile: TravelProfile) -> Itinerary:
    """
    Convertit le JSON (str) en Itinerary (Pydantic) en effectuant une validation minimale.
    """
    obj = json.loads(json_text)
    days_out: List[ItineraryDay] = []

    for day in obj.get("days", []):
        morning: List[ItineraryActivity] = []
        for a in day.get("morning", []):
            morning.append(
                ItineraryActivity(
                    name=a.get("name", ""),
                    category=a.get("category"),
                    description=a.get("description"),
                    budget=a.get("budget"),
                    best_time=a.get("best_time"),
                    tips=a.get("tips"),
                    source_id=a.get("source_id"),
                    source_city=a.get("source_city"),
                )
            )

        afternoon: List[ItineraryActivity] = []
        for a in day.get("afternoon", []):
            afternoon.append(
                ItineraryActivity(
                    name=a.get("name", ""),
                    category=a.get("category"),
                    description=a.get("description"),
                    budget=a.get("budget"),
                    best_time=a.get("best_time"),
                    tips=a.get("tips"),
                    source_id=a.get("source_id"),
                    source_city=a.get("source_city"),
                )
            )

        evening: List[ItineraryActivity] = []
        for a in day.get("evening", []):
            evening.append(
                ItineraryActivity(
                    name=a.get("name", ""),
                    category=a.get("category"),
                    description=a.get("description"),
                    budget=a.get("budget"),
                    best_time=a.get("best_time"),
                    tips=a.get("tips"),
                    source_id=a.get("source_id"),
                    source_city=a.get("source_city"),
                )
            )

        day_obj = ItineraryDay(
            day_number=int(day.get("day_number", 0)),
            morning=morning,
            afternoon=afternoon,
            evening=evening,
        )
        days_out.append(day_obj)

    itinerary = Itinerary(
        city=obj.get("city") or profile.city,
        duration_days=profile.duration_days,
        profile=profile,
        days=days_out,
        notes=obj.get("notes"),
    )
    return itinerary


# ---------- LLM call (Hugging Face) ----------

def generate_itinerary(profile: TravelProfile, max_docs: int = 30) -> Itinerary:
    """
    Fonction principale : génère un Itinerary à partir d'un TravelProfile.

    Étapes :
    - récupère candidats via FAISS (RAG)
    - construit un prompt détaillé
    - appelle le LLM Hugging Face
    - extrait et parse le JSON renvoyé
    - retourne un objet Itinerary (Pydantic)
    """
    # 1) récupérer les docs candidats
    docs = get_candidate_places(profile, max_docs=max_docs)
    if not docs:
        raise ValueError(
            "Aucune donnée trouvée pour le profil donné. Vérifie la base de connaissances (CSV) et l'index FAISS."
        )

    # 2) formater pour prompt
    places_text = format_places_for_prompt(docs)

    # 3) construire prompt
    prompt = build_itinerary_prompt(profile, places_text)

    # 4) invoquer LLM Hugging Face (recyclage de get_llm() défini dans qa_chain)
    llm = get_llm()

    try:
        # API moderne : invoke()
        raw_text = llm.invoke(prompt)
    except TypeError:
        # fallback : certains wrappers supportent l'appel direct
        raw_text = llm(prompt)

    # Normaliser en string si nécessaire
    if not isinstance(raw_text, str):
        # Certains LLMs pourraient renvoyer un objet avec .content ou autre
        if hasattr(raw_text, "content"):
            raw_text = raw_text.content
        else:
            raw_text = str(raw_text)

    # 5) extraire JSON
    json_str = extract_json_from_text(raw_text or "")
    if not json_str:
        raise ValueError(
            "Impossible d'extraire un JSON valide de la réponse du LLM.\n"
            f"Réponse brute:\n{raw_text}"
        )

    # 6) parser en Itinerary
    try:
        itinerary = parse_itinerary_json(json_str, profile)
    except Exception as e:
        raise ValueError(
            f"Le JSON extrait est invalide ou inattendu: {e}\n"
            f"JSON extrait (début): {json_str[:1000]}"
        )

    return itinerary


# ---------- Usage example (pour tests / notebooks) ----------

if __name__ == "__main__":
    # Test rapide (nécessite FAISS index + variables HF configurées)
    sample_profile = TravelProfile(
        city="Marrakech",
        duration_days=3,
        budget="medium",
        interests=["culture", "gastronomy"],
        constraints="éviter trop de marche",
        language="fr",
    )
    iti = generate_itinerary(sample_profile, max_docs=20)
    print(iti.json(indent=2, ensure_ascii=False))
