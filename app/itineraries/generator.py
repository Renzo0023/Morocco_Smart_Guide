# app/itineraries/generator.py

from __future__ import annotations

import json
from typing import List, Optional

from datetime import time, timedelta, datetime

# Compat Documents
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from huggingface_hub import InferenceClient

from app.config import HF_API_KEY, LLM_MODEL_NAME
from app.itineraries.models import (
    TravelProfile,
    Itinerary,
    ItineraryDay,
    ItineraryActivity,
)
from app.rag.vectorstore import get_retriever


# ---------- Helpers RAG ----------

def build_retriever_query(profile: TravelProfile) -> str:
    """
    Construit une requête textuelle pour interroger FAISS selon le profil.
    """
    interests = ", ".join(profile.interests) if profile.interests else "général"
    constraints = profile.constraints or "aucune contrainte particulière"
    city_clause = f"Ville: {profile.city}." if profile.city else "Multi-ville."

    return (
        f"{city_clause} Séjour de {profile.duration_days} jours. "
        f"Budget: {profile.budget}. Centres d'intérêt: {interests}. "
        f"Contraintes: {constraints}."
    )


def get_candidate_places(profile: TravelProfile, max_docs: int = 30) -> List[Document]:
    """
    Utilise le retriever FAISS pour obtenir des documents candidats.

    Compatible avec les nouvelles versions de LangChain :
    - tente get_relevant_documents()
    - sinon tente invoke()
    - sinon descend au niveau vectorstore.similarity_search()
    """
    retriever = get_retriever(k=max_docs)
    query = build_retriever_query(profile)

    docs: List[Document] = []

    # 1) Nouveau style: get_relevant_documents
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
    else:
        # 2) API de BaseRetriever (invoke)
        try:
            result = retriever.invoke(query)
            if isinstance(result, list):
                docs = result
            else:
                docs = [result]
        except Exception:
            # 3) fallback bas niveau : vecteur FAISS sous-jacent
            vectorstore = getattr(retriever, "vectorstore", None)
            if vectorstore is not None and hasattr(vectorstore, "similarity_search"):
                docs = vectorstore.similarity_search(query, k=max_docs)
            else:
                raise RuntimeError(
                    "Impossible de récupérer des documents avec le retriever : "
                    "ni get_relevant_documents, ni invoke, ni vectorstore.similarity_search disponibles."
                )

    # Filtrer par city metadata si demandé
    if profile.city:
        filtered = []
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
                    filtered.append(d)
            else:
                filtered.append(d)
        docs = filtered

    return docs[:max_docs]


def format_places_for_prompt(docs: List[Document], max_chars: int = 20000) -> str:
    """
    Transforme les documents en un grand bloc de texte lisible par le LLM.
    Tronque si trop long.
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



def parse_place_from_doc(doc: Document) -> dict:
    """
    Extrait une structure exploitable depuis un Document FAISS.
    """
    meta = doc.metadata or {}

    lat = meta.get("lat")
    lon = meta.get("lon")

    maps_url = generate_maps_url(meta.get("name", ""), meta.get("city", ""))

    return {
        "id": meta.get("id"),
        "name": meta.get("name"),
        "city": meta.get("city"),
        "category": meta.get("category"),
        "description": doc.page_content,
        "budget": meta.get("budget"),
        "best_time": meta.get("best_time"),
        "duration": float(meta.get("duration_hours", 1.5)),
        "tips": meta.get("tips"),
        "maps_url": maps_url,
    }


def allocate_times(
    activities: List[dict],
    start_hour: int
) -> List[dict]:
    """
    Assigne des horaires continus aux activités.
    """
    current = datetime(2000, 1, 1, start_hour, 0)

    for a in activities:
        duration = float(a.get("duration", 1.5))
        a["start_time"] = current.strftime("%H:%M")
        current += timedelta(hours=duration)
        a["end_time"] = current.strftime("%H:%M")

    return activities


def build_time_based_plan(
    places: List[dict],
    duration_days: int
) -> List[dict]:
    """
    Organise les lieux en jours / matin / après-midi / soir
    en respectant des budgets temps réalistes.
    """

    TIME_SLOTS = {
        "morning": 4.0,
        "afternoon": 4.0,
        "evening": 3.0,
    }

    days = [
        {
            "day_number": d + 1,
            "morning": [],
            "afternoon": [],
            "evening": [],
            "remaining": TIME_SLOTS.copy()
        }
        for d in range(duration_days)
    ]

    # Prioriser les lieux longs en premier
    places = sorted(places, key=lambda x: x["duration"], reverse=True)

    day_index = 0

    for place in places:
        placed = False
        preferred = place.get("best_time")

        while day_index < duration_days and not placed:
            day = days[day_index]

            slots = [preferred] if preferred in TIME_SLOTS else TIME_SLOTS.keys()

            for slot in slots:
                if place["duration"] <= day["remaining"][slot]:
                    day[slot].append(place)
                    day["remaining"][slot] -= place["duration"]
                    placed = True
                    break

            if not placed:
                day_index += 1

        if day_index >= duration_days:
            break

    preferred = place.get("best_time")

    slots = (
        [preferred]
        if preferred in TIME_SLOTS and day["remaining"][preferred] >= place["duration"]
        else TIME_SLOTS.keys()
    )

    # Assignation des horaires
    for d in days:
        d["morning"] = allocate_times(d["morning"], start_hour=9)
        d["afternoon"] = allocate_times(d["afternoon"], start_hour=14)
        d["evening"] = allocate_times(d["evening"], start_hour=18)

        d.pop("remaining", None)

    return days



def build_itinerary_prompt(profile: TravelProfile, places_text: str) -> str:
    """
    Construire un prompt explicite demandant une sortie JSON strictement formatée.
    """
    interests = ", ".join(profile.interests) if profile.interests else "général"
    constraints = profile.constraints or "aucune contrainte particulière"

    prompt = f"""
Tu es un expert en tourisme local et planification de voyages au Maroc.

Contrainte utilisateur:
- Ville principale: {profile.city or 'Multi-villes (pas de ville spécifiée)'}
- Durée (jours): {profile.duration_days}
- Budget: {profile.budget}
- Centres d'intérêt: {interests}
- Contraintes: {constraints}
- Langue souhaitée: {profile.language}


Voici un planning pré-organisé par un algorithme de planification
tenant compte de la durée des visites et des moments de la journée.
Tu DOIS respecter strictement cette structure.

Planning JSON pré-calculé (NE PAS MODIFIER):
{places_text}

Pour chaque activité du planning:
- name → name
- category → category
- description → description
- budget → budget
- best_time → best_time
- tips → tips
- id → source_id
- city → source_city

Instructions:
1) Enrichis et améliore ce planning sans modifier la structure.
2) Chaque jour doit contenir au minimum une activité le matin et une l'après-midi; le soir est optionnel.
3) Respecte le budget et les centres d'intérêt autant que possible.
4) N'ajoute pas de lieux qui ne figurent pas dans la liste ci-dessus.
5) Réponds STRICTEMENT en JSON valide avec la structure suivante:

{{
  "city": "{profile.city or ''}",
  "days": [
    {{
      "day_number": 1,
      "morning": [
        {{
          "name": "...",
          "start_time": "HH:MM",
          "end_time": "HH:MM",
          "category": "...",
          "description": "...",
          "budget": "...",
          "best_time": "...",
          "tips": "...",
          "source_id": "...",
          "source_city": "...",
          "maps_url": "https://www.google.com/maps/..."
        }}
      ],
      "afternoon": [...],
      "evening": [...]
    }}
  ]
}}

IMPORTANT:
- Ne fournis aucun texte hors du JSON.
- Si tu dois mentionner un manque d'information, place le message dans un champ "notes" au niveau du JSON.
Réponds dans la langue demandée: {profile.language}.
- Ne change pas les jours.
- Ne déplace pas une activité vers un autre moment.
- Ne supprime pas d’activité.
- N’ajoute aucun lieu.
- Améliore uniquement les descriptions, conseils et cohérence narrative.
- Ne modifie jamais les champs start_time et end_time.
- Les horaires sont déjà calculés par le système.
"""
    return prompt


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Tente d'extraire le premier objet JSON complet trouvé dans `text`.
    Renvoie la string JSON ou None.
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
                return text[start: i + 1]
    return None


def generate_maps_url(name: str, city: str) -> str:
    base_url = "https://www.google.com/maps/search/"
    query = f"{name}, {city}".replace(" ", "+")
    return base_url + query


def parse_itinerary_json(json_text: str, profile: TravelProfile) -> Itinerary:
    """
    Convertit le JSON (str) en Itinerary (Pydantic).
    """
    obj = json.loads(json_text)
    days_out: List[ItineraryDay] = []

    for day in obj.get("days", []):
        morning = []
        for a in day.get("morning", []):
            morning.append(
                ItineraryActivity(
                    name=a.get("name", ""),
                    start_time=a.get("start_time"),
                    end_time=a.get("end_time"),
                    category=a.get("category"),
                    description=a.get("description"),
                    budget=a.get("budget"),
                    best_time=a.get("best_time"),
                    tips=a.get("tips"),
                    source_id=a.get("source_id"),
                    source_city=a.get("source_city"),
                    maps_url=generate_maps_url(a["name"], a.get("city") or a.get("source_city") or profile.city),
                )
            )

        afternoon = []
        for a in day.get("afternoon", []):
            afternoon.append(
                ItineraryActivity(
                    name=a.get("name", ""),
                    start_time=a.get("start_time"),
                    end_time=a.get("end_time"),
                    category=a.get("category"),
                    description=a.get("description"),
                    budget=a.get("budget"),
                    best_time=a.get("best_time"),
                    tips=a.get("tips"),
                    source_id=a.get("source_id"),
                    source_city=a.get("source_city"),
                    maps_url=generate_maps_url(a["name"], a.get("city") or a.get("source_city") or profile.city),
                )
            )

        evening = []
        for a in day.get("evening", []):
            evening.append(
                ItineraryActivity(
                    name=a.get("name", ""),
                    start_time=a.get("start_time"),
                    end_time=a.get("end_time"),
                    category=a.get("category"),
                    description=a.get("description"),
                    budget=a.get("budget"),
                    best_time=a.get("best_time"),
                    tips=a.get("tips"),
                    source_id=a.get("source_id"),
                    source_city=a.get("source_city"),
                    maps_url=generate_maps_url(a["name"], a.get("city") or a.get("source_city") or profile.city),
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
    )
    return itinerary


# ---------- Appel direct du LLM Hugging Face ----------
# ---------- Appel direct du LLM Hugging Face ----------
def call_hf_llm(prompt: str) -> str:
    """
    Appelle le modèle Hugging Face via l'API chat_completion
    compatible avec InferenceClient.
    Extraction robuste adaptée à tous les formats HF.
    """
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY manquant dans .env")

    client = InferenceClient(model=LLM_MODEL_NAME, token=HF_API_KEY)

    resp = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.5,
        top_p=0.9
    )

    # ----------------------------------------
    # Extraction compatible tous modèles HF
    # ----------------------------------------
    try:
        if hasattr(resp, "choices") and resp.choices:
            choice = resp.choices[0]

            # 1) Format Mistral/Llama HF
            if hasattr(choice, "text") and isinstance(choice.text, str):
                return choice.text

            # 2) Format OpenAI-like
            if hasattr(choice, "message") and isinstance(choice.message, dict):
                if "content" in choice.message:
                    return choice.message["content"]
    except Exception:
        pass

    # 3) Format direct HuggingFace
    if hasattr(resp, "generated_text"):
        return resp.generated_text

    # 4) Fallback
    return str(resp)




def generate_itinerary(profile: TravelProfile, max_docs: int = 30) -> Itinerary:
    """
    Fonction principale exposée: génère un Itinerary à partir d'un TravelProfile.
    """
    # 1) récupérer les docs candidats
    docs = get_candidate_places(profile, max_docs=max_docs)
    if not docs:
        raise ValueError(
            "Aucune donnée trouvée pour le profil donné. Vérifie la base de connaissances."
        )

    # 2) formater pour prompt
    # places_text = format_places_for_prompt(docs)
    # 3) construire prompt
    # prompt = build_itinerary_prompt(profile, places_text)

    # 2) transformer les docs en lieux exploitables
    places = [parse_place_from_doc(d) for d in docs]

    # 3) construire un planning temps-réel
    planned_days = build_time_based_plan(
        places=places,
        duration_days=profile.duration_days
   )

    # 4) injecter le planning (et non la liste brute)
    planning_text = json.dumps(planned_days, ensure_ascii=False, indent=2)

    prompt = build_itinerary_prompt(profile, planning_text)

    # 4) invoquer le LLM Hugging Face directement
    try:
        raw_text = call_hf_llm(prompt)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'appel LLM: {e}")

    # 5) extraire JSON robuste
    json_str = extract_json_from_text(raw_text or "")
    if not json_str:
        raise ValueError(
            f"Impossible d'extraire un JSON de la réponse du LLM. Réponse brute:\n{raw_text}"
        )

    # 6) parser en Itinerary (Pydantic)
    try:
        itinerary = parse_itinerary_json(json_str, profile)
    except Exception as e:
        raise ValueError(
            f"Le JSON extrait est invalide ou inattendu: {e}\nJSON_Str: {json_str[:1000]}"
        )

    return itinerary


# ---------- Usage example (pour notebooks) ----------
if __name__ == "__main__":
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
