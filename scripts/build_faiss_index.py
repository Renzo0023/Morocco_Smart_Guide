"""
scripts/build_faiss_index.py

Script pour construire et sauvegarder l'index FAISS à partir
des fichiers CSV de lieux touristiques (data/*_places.csv).

⚠️ Prérequis :
- Avoir les CSV dans le dossier data/
- Avoir configuré .env (EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, etc.)
- Avoir installé les dépendances (faiss-cpu, sentence-transformers, langchain, ...)
"""

from app.rag.vectorstore import build_and_save_faiss_index
from app.data.loader import load_places


def main():
    try:
        places = load_places()
        print(f"[DATA] {len(places)} lieux chargés depuis les CSV.")
    except Exception as e:
        print(f"[DATA] Erreur lors du chargement des lieux : {e}")
        return

    print("[FAISS] Construction de l'index...")
    try:
        build_and_save_faiss_index(places)
        print("[FAISS] Index construit et sauvegardé avec succès ✅")
    except Exception as e:
        print(f"[FAISS] Erreur lors de la construction de l'index : {e}")


if __name__ == "__main__":
    main()
