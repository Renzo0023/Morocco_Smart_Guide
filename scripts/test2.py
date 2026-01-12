from app.rag.vectorstore import get_retriever

def debug_city(city: str, k: int = 50):
    print(f"\n🔎 Diagnostic FAISS pour la ville : {city}\n")

    retriever = get_retriever(k=k, city=city, category=None)

    query = f"Lieux touristiques à {city}"
    
    if hasattr(retriever, "get_relevant_documents"):
        docs = retriever.get_relevant_documents(query)
    else:
        docs = retriever.invoke(query)

    print(f"Documents retournés: {len(docs)}\n")

    if not docs:
        print("❌ AUCUN document trouvé pour cette ville.")
        return

    cities = {}
    categories = {}
    broken = 0

    for d in docs:
        meta = d.metadata or {}
        c = meta.get("city")
        cat = meta.get("category")

        if not c:
            broken += 1

        cities[c] = cities.get(c, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1

    print("📍 Répartition par ville:")
    for k, v in cities.items():
        print(f"  {k}: {v}")

    print("\n📂 Catégories:")
    for k, v in categories.items():
        print(f"  {k}: {v}")

    print(f"\n⚠️ Documents sans city: {broken}")

    print("\n📄 Exemples:")
    for i, d in enumerate(docs[:5]):
        meta = d.metadata
        print(f"\n--- {i+1}")
        print("Nom:", meta.get("name"))
        print("Ville:", meta.get("city"))
        print("Catégorie:", meta.get("category"))
        print("Durée:", meta.get("duration_hours"))
        print("Budget:", meta.get("budget"))
        print("Contenu:", d.page_content[:200], "...")

if __name__ == "__main__":
    debug_city("Tanger")
