# app/rag/embeddings.py
from langchain.embeddings import OpenAIEmbeddings
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


_embeddings_cache = None

def get_embeddings():
    """
    Retourne un objet embeddings OpenAI.
    Utilise un cache interne pour ne pas recréer l'objet à chaque appel.
    """
    global _embeddings_cache

    if _embeddings_cache is None:
        _embeddings_cache = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY
        )
    return _embeddings_cache

