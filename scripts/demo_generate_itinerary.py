from pprint import pprint

from app.itineraries.models import TravelProfile
from app.itineraries.generator import generate_itinerary


def print_itinerary_pretty(itinerary):
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
        print("\n💡 Notes générales :")
        print(itinerary.notes)
        print()

    print("\n===== JSON complet (pour debug / API) =====")
    print(itinerary.model_dump_json(indent=2, ensure_ascii=False))


def main():
    profile = TravelProfile(
        city="Marrakech",
        duration_days=3,
        budget="medium",
        interests=["culture", "gastronomy", "shopping"],
        constraints="éviter trop de marche",
        language="fr",
    )

    print("Profil de test :")
    pprint(profile.model_dump())
    print("\nGénération de l'itinéraire...\n")

    try:
        itinerary = generate_itinerary(profile, max_docs=30)
    except Exception as e:
        print(f"❌ Erreur lors de la génération de l'itinéraire : {e}")
        return

    print_itinerary_pretty(itinerary)


if __name__ == "__main__":
    main()
