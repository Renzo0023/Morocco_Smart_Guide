# app/rag/vectorstore.py

from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.vectorstores import FAISS

from app.rag.embeddings import get_embeddings
from app.data.loader import load_places, to_documents

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "faiss_index"


def build_faiss_index(documents: Optional[List[Document]] = None) -> FAISS:
    if documents is None:
        places = load_places()
        documents = to_documents(places)

    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)


def build_and_save_faiss_index(save_dir: Path | None = None):
    save_dir = save_dir or INDEX_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    vs = build_faiss_index()
    vs.save_local(str(save_dir))
    print(f"[FAISS] Index construit et sauvegardé dans {save_dir}")


def load_faiss_index(load_dir: Path | None = None) -> FAISS:
    load_dir = load_dir or INDEX_DIR
    if not load_dir.exists():
        raise FileNotFoundError(
            f"Index introuvable dans {load_dir}. Lancer build_and_save_faiss_index()."
        )
    embeddings = get_embeddings()
    return FAISS.load_local(str(load_dir), embeddings)


# Nouveau : retriever avec filtres NOTABLES
def get_retriever(k: int = 5, city: Optional[str] = None, category: Optional[str] = None):
    vs = load_faiss_index()

    search_kwargs = {"k": k}

    # Filtrage via metadata pour multi-villes
    metadata_filter = {}
    if city:
        metadata_filter["city"] = city
    if category:
        metadata_filter["category"] = category

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    return vs.as_retriever(search_kwargs=search_kwargs)

