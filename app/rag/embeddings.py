# app/rag/embeddings.py

from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL_NAME

_embeddings_cache = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Retourne un objet HuggingFaceEmbeddings basé sur un modèle open-source.

    Le modèle utilisé est défini dans config.py, par exemple :
      EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    Un cache interne (_embeddings_cache) est utilisé pour éviter de recharger 
    le modèle à chaque appel, car le chargement peut être coûteux.
    """
    global _embeddings_cache

    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},          # "cuda" si tu as un GPU
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings_cache
