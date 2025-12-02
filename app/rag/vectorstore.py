# app/rag/vectorstore.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

# Compatibilité LangChain (ancienne / nouvelle version)
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain.vectorstores import FAISS

from app.rag.embeddings import get_embeddings
from app.data.loader import load_places, to_documents
from app.config import FAISS_INDEX_PATH  # défini dans config.py


# Dossier où l'index FAISS est sauvegardé
INDEX_DIR = Path(FAISS_INDEX_PATH)


def build_faiss_index(documents: Optional[List[Document]] = None) -> FAISS:
    """
    Construit un index FAISS en mémoire à partir :
      - d'une liste de Documents fournie, ou
      - (par défaut) des lieux chargés depuis les CSV du dossier data/.

    Retourne un objet FAISS utilisable comme VectorStore.
    """
    if documents is None:
        places = load_places()
        documents = to_documents(places)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def build_and_save_faiss_index(save_dir: Path | None = None) -> None:
    """
    Construit l'index FAISS et le sauvegarde dans un dossier (par défaut INDEX_DIR).

    À appeler une première fois (ou à chaque mise à jour des données) via un script :
      python scripts/build_faiss_index.py
    """
    save_dir = save_dir or INDEX_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    vs = build_faiss_index()
    vs.save_local(str(save_dir))

    print(f"[FAISS] Index construit et sauvegardé dans {save_dir}")


def load_faiss_index(load_dir: Path | None = None) -> FAISS:
    """
    Charge l'index FAISS depuis le disque.

    Si l'index n'existe pas, lève une erreur avec un message clair.
    """
    load_dir = load_dir or INDEX_DIR

    if not load_dir.exists():
        raise FileNotFoundError(
            f"Index introuvable dans {load_dir}. "
            f"Lancer d'abord build_and_save_faiss_index()."
        )

    embeddings = get_embeddings()

    # Compat LangChain : certaines versions exigent allow_dangerous_deserialization=True
    try:
        vectorstore = FAISS.load_local(str(load_dir), embeddings)
    except TypeError:
        vectorstore = FAISS.load_local(
            str(load_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return vectorstore


def get_retriever(
    k: int = 5,
    city: Optional[str] = None,
    category: Optional[str] = None,
):
    """
    Retourne un retriever basé sur l'index FAISS.

    - k : nombre de documents à récupérer
    - city : filtre sur la ville (metadata["city"])
    - category : filtre sur la catégorie (metadata["category"])

    Ces metadata viennent de app.data.loader.place_to_document()
    """
    vs = load_faiss_index()

    search_kwargs: dict = {"k": k}

    # Filtrage via metadata pour gérer le multi-villes / multi-catégories
    metadata_filter: dict = {}
    if city:
        metadata_filter["city"] = city
    if category:
        metadata_filter["category"] = category

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vs.as_retriever(search_kwargs=search_kwargs)
    return retriever
