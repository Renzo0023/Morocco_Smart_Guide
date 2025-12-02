# app/itineraries/generator.py
import json
import re
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage

from app.itineraries.models import (
    TravelProfile,
    Itinerary,
    ItineraryDay,
    ItineraryActivity,
)
# NOTE: we import get_retriever from the rag module.
# If your P2 implemented get_retriever() in app.rag.vectorstore.py, change the import accordingly.
try:
    from app.rag.vectorstore import get_retriever  # preferred if vectorstore provides it
except Exception:
    try:
        from app.rag.qa_chain import get_retriever  # fallback
    except Exception:
        # If neither exists, raise informative error at import time.
        raise ImportError(
            "Missing get_retriever() in app.rag.vectorstore or app.rag.qa_chain. "
            "P2 doit exposer une fonction get_retriever() qui renvoie un LangChain retriever."
        )

from app.config import OPENAI_MODEL, OPENAI_API_KEY  # assure-toi que config.py existe


# ---------- Helpers ----------

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
    Si les documents ont la metadata 'city' et que profile.city est renseigné,
    on filtre pour ne garder que ceux de la ville souhaitée.
    """
    retriever = get_retriever()
    query = build_retriever_query(profile)

    # utiliser la méthode standard du retriever pour obtenir les docs
    # Certains retrievers fournissent `get_relevant_documents(query)` ; sinon `retrieve`/`similarity_search`
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
    elif hasattr(retriever, "get_relevant_documents_by_query"):
        docs = retriever.get_relevant_documents_by_query(query)
    else:
        # fallback : essayer similarity_search
        docs = retriever.similarity_search(query, k=max_docs if max_docs else 10)

    # Filtrer par city metadata si demandé
    if profile.city:
        filtered = []
        for d in docs:
            meta_city = None
            if isinstance(d.metadata, dict):
                meta_city = d.metadata.get("city") or d.metadata.get("location") or d.metadata.get("source_city")
                if meta_city and isinstance(meta_city, str):
                    if profile.city.lower() in meta_city.lower():
                        filtered.append(d)
                else:
                    # keep if no metadata city (conservative)
                    filtered.append(d)
            else:
                filtered.append(d)
        docs = filtered

    return docs[:max_docs]


def format_places_for_prompt(docs: List[Document], max_chars: int = 20000) -> str:
    """
    Transforme les documents en un grand bloc de texte lisible par le LLM.
    tronque si trop long (max_chars).
    """
    parts = []
    for d in docs:
        meta = d.metadata or {}
        name = meta.get("name") or meta.get("title") or (d.page_content[:50] + "...")
        city = meta.get("city") or meta.get("location") or ""
        category = meta.get("category") or ""
        budget = meta.get("budget") or ""
        best_time = meta.get("best_time") or ""
        # page_content contient la description préparée par P1 -> utile
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
    Construire un prompt explicite demandant une sortie JSON strictement formatée.
    """
    interests = ", ".join(profile.interests) if profile.interests else "général"
    constraints = profile.constraints or "aucune contrainte particulière"

    prompt = f"""
Tu es un expert en tourisme local et planification de voyages.

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
          "tips": "..."
        }}
      ],
      "afternoon": [...],
      "evening": [...]
    }}
  ]
}}

IMPORTANT: Ne fournis aucun texte hors du JSON. Si tu dois mentionner un manque d'information, place le message dans un champ "notes" au niveau du JSON.
"""
    return prompt


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Tente d'extraire le premier objet JSON complet trouvé dans `text`.
    Renvoie la string JSON ou None.
    Méthode robuste: trouve la première '{' et la dernière '}' correspondante.
    """
    # Cherche premier '{'
    start = text.find("{")
    if start == -1:
        return None

    # Balancer accolades : parcourt et trouve la position finale qui équilibre
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_itinerary_json(json_text: str, profile: TravelProfile) -> Itinerary:
    """
    Convertit le JSON (str) en Itinerary (Pydantic) en effectuant une validation minimale.
    """
    obj = json.loads(json_text)
    days_out = []
    for day in obj.get("days", []):
        morning = []
        for a in day.get("morning", []):
            morning.append(
                ItineraryActivity(
                    name=a.get("name", ""),
                    category=a.get("category"),
                    description=a.get("description"),
                    budget=a.get("budget"),
                    best_time=a.get("best_time"),
                    tips=a.get("tips"),
                    source_id=a.get("source_id'),
                    source_city=a.get("source_city"),
                )
            )
        afternoon = []
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
        evening = []
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
    )
    return itinerary


# ---------- LLM call ----------

def get_llm():
    """
    Instancie le ChatOpenAI.
    Assure-toi que app.config fournit OPENAI_API_KEY et OPENAI_MODEL.
    """
    # On utilise langchain.chat_models.ChatOpenAI (moderne)
    # temperature modérée pour varier sans perdre cohérence
    chat = ChatOpenAI(
        model_name=OPENAI_MODEL,
        temperature=0.5,
        openai_api_key=OPENAI_API_KEY
    )
    return chat


def generate_itinerary(profile: TravelProfile, max_docs: int = 30) -> Itinerary:
    """
    Fonction principale exposée: génère un Itinerary à partir d'un TravelProfile.

    Étapes :
    - récupère candidats via FAISS (P2)
    - construit prompt
    - appelle LLM
    - extrait et parse le JSON
    - retourne Itinerary (Pydantic)
    """
    # 1) récupérer les docs candidats
    docs = get_candidate_places(profile, max_docs=max_docs)
    if not docs:
        # fallback minimal si pas de documents
        raise ValueError("Aucune donnée trouvée pour le profil donné. Vérifie la base de connaissances.")

    # 2) formater pour prompt
    places_text = format_places_for_prompt(docs)

    # 3) construire prompt
    prompt = build_itinerary_prompt(profile, places_text)

    # 4) invoquer LLM (sous forme de chat)
    llm = get_llm()
    system_msg = SystemMessage(content="Tu es un assistant expert en planification de voyages.")
    human_msg = HumanMessage(content=prompt)

    try:
        response = llm([system_msg, human_msg])
        # Selon la version, response peut renvoyer un object avec .content ou un str
        raw_text = None
        if isinstance(response, str):
            raw_text = response
        else:
            # ChatOpenAI retourne souvent un object avec .content accessible via response.content
            # mais en LangChain le return type peut être un ChatResult -> response.generations...
            # Tentons plusieurs accès
            if hasattr(response, "content"):
                raw_text = response.content
            elif hasattr(response, "generations"):
                # response.generations : list[List[ChatGeneration]]
                try:
                    raw_text = response.generations[0][0].text
                except Exception:
                    raw_text = str(response)
            else:
                raw_text = str(response)

    except Exception as e:
        raise RuntimeError(f"Erreur lors de l'appel LLM: {e}")

    # 5) extraire JSON robuste
    json_str = extract_json_from_text(raw_text or "")
    if not json_str:
        # tentatives de nettoyage simples avant d'abandonner
        # Parfois l'IA renvoie du texte avec "Réponse:\n{...}\n" -> on tente d'extraire d'abord '{'...' }'
        # Si extraction échoue, on échoue proprement.
        raise ValueError(f"Impossible d'extraire un JSON de la réponse du LLM. Réponse brute:\n{raw_text}")

    # 6) parser en Itinerary (Pydantic)
    try:
        itinerary = parse_itinerary_json(json_str, profile)
    except Exception as e:
        raise ValueError(f"Le JSON extrait est invalide ou inattendu: {e}\nJSON_Str: {json_str[:1000]}")

    return itinerary


# ---------- Usage example (pour notebooks) ----------
if __name__ == "__main__":
    # petit test rapide (à lancer dans un environnement avec clés et index FAISS)
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
