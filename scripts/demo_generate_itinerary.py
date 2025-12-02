"""
scripts/demo_generate_itinerary.py

Petit script de démonstration pour tester la génération d'itinéraire
sans passer par l'interface Streamlit ni l'API FastAPI.

⚠️ Prérequis :
- Avoir construit l'index FAISS :
    python -m scripts.build_faiss_index
- Avoir configuré le fichier .env avec :
    HF_API_KEY, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, etc.
"""

from pprint import pprint

from app.itineraries.models import TravelProfile
from app.itineraries.generator import generate_itinerary


def print_itinerary_pretty(itinerary):
    """
    Affiche un itinéraire dans un format lisible en console.
    """
    print("=" * 60)
    print(f" Itinéraire pour {itinerary.city} - {itinerary.duration_days} jours")
    print("=" * 60)
    print()

    for day in itinerary.days:
        print(f"🗓️  Jour {day.day_number}")
        print("  🌅 Matin :")
        if day.morning:
            for act in day.morning:
                print(f"    - {act.name} ({act.category})")
                if act.description:
                    print(f"      {act.description}")
        else:
            print("    (aucune activité prévue)")

        print("  🌞 Après-midi :")
        if day.afternoon:
            for act in day.afternoon:
                print(f"    - {act.name} ({act.category})")
                if act.description:
                    print(f"      {act.description}")
        else:
            print("    (aucune activité prévue)")

        print("  🌙 Soir :")
        if day.evening:
            for act in day.evening:
                print(f"    - {act.name} ({act.category})")
                if act.description:
                    print(f"      {act.description}")
        else:
            print("    (aucune activité prévue)")

        print("-" * 60)

    if getattr(itinerary, "notes", None):
        print()
        print("💡 Notes générales :")
        print(itinerary.notes)
        print()

    print()
    print("===== JSON complet (pour debug / API) =====")
    print(itinerary.json(indent=2, ensure_ascii=False))


def main():
    """
    Crée un profil de test et génère un itinéraire complet.
    Modifie les valeurs ci-dessous pour tester différents scénarios.
    """
    profile = TravelProfile(
        city="Marrakech",              # ou None pour multi-villes si tu as plusieurs CSV
        duration_days=3,
        budget="medium",               # "low" | "medium" | "high"
        interests=["culture", "gastronomy", "shopping"],
        constraints="éviter trop de marche",
        language="fr",
    )

    print("Profil de test :")
    pprint(profile.dict())
    print("\nGénération de l'itinéraire... (cela peut prendre quelques secondes)\n")

    try:
        itinerary = generate_itinerary(profile, max_docs=30)
    except Exception as e:
        print(f"❌ Erreur lors de la génération de l'itinéraire : {e}")
        return

    print_itinerary_pretty(itinerary)


if __name__ == "__main__":
    main()
