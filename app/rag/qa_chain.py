# app/rag/qa_chain.py

from __future__ import annotations
from typing import Dict, Any, List

from huggingface_hub import InferenceClient

from app.config import HF_API_KEY, LLM_MODEL_NAME
from app.rag.vectorstore import get_retriever


# ============================================================
#  MÉMOIRE MAISON (remplace ConversationBufferMemory)
# ============================================================

class SimpleMemory:
    """
    Mémoire minimale compatible :
    - history = liste de {"role": "user"/"assistant", "content": "..."}
    """

    def __init__(self):
        self.chat_history: List[Dict[str, str]] = []

    def add_user_message(self, text: str):
        self.chat_history.append({"role": "user", "content": text})

    def add_ai_message(self, text: str):
        self.chat_history.append({"role": "assistant", "content": text})

    def get_history_as_text(self) -> str:
        formatted = ""
        for msg in self.chat_history:
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n"
        return formatted


# ============================================================
#  Appel direct du LLM Hugging Face (InferenceClient)
# ============================================================
# ============================================================
#  Appel direct du LLM Hugging Face (InferenceClient)
# ============================================================

def call_hf_llm(prompt: str) -> str:
    if not HF_API_KEY:
        raise ValueError("HF_API_KEY manquant dans .env")

    client = InferenceClient(model=LLM_MODEL_NAME, token=HF_API_KEY)

    resp = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.4,
        top_p=0.9,
    )

    # -------------------------------
    # Extraction compatible Mistral
    # -------------------------------
    try:
        # Format Mistral, Llama HF : {"choices": [{"text": "..."}]}
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice = resp.choices[0]

            if hasattr(choice, "text") and isinstance(choice.text, str):
                return choice.text  # <-- MISTRAL v0.2 utilise ça !

            # Format OpenAI-like
            if hasattr(choice, "message") and isinstance(choice.message, dict):
                if "content" in choice.message:
                    return choice.message["content"]
    except Exception:
        pass

    # Format direct HuggingFace ("generated_text")
    if hasattr(resp, "generated_text"):
        return resp.generated_text

    return str(resp)



# ============================================================
#  CHAÎNE RAG CONVERSATIONNELLE (implémentation maison)
# ============================================================

class SimpleRAGConversationChain:
    """
    Compatible avec ton ancien code :
        result = chain({"question": "..."} )
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.memory = SimpleMemory()

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs.get("question")
        if not question:
            raise ValueError("La clé 'question' est requise.")

        # 1) Ajouter question à la mémoire utilisateur
        self.memory.add_user_message(question)

        # 2) RAG : documents pertinents
        docs = self.retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        # 3) Historique formaté
        history_text = self.memory.get_history_as_text()

        # 4) Prompt final
        prompt = (
            "Tu es un expert du tourisme au Maroc. Réponds uniquement en te basant sur le contexte.\n"
            "Si l'information n'est pas disponible, dis-le.\n\n"
            "=== HISTORIQUE ===\n"
            f"{history_text}\n\n"
            "=== CONTEXTE ===\n"
            f"{context}\n\n"
            "=== QUESTION ===\n"
            f"{question}\n\n"
            "Réponds clairement et utilement.\n"
        )

        # 5) Appel du modèle Hugging Face
        answer = call_hf_llm(prompt)

        # 6) Ajouter réponse dans la mémoire
        self.memory.add_ai_message(answer)

        return {
            "answer": answer,
            "source_documents": docs,
        }


# ============================================================
#  Fonction publique pour FastAPI
# ============================================================

def get_rag_conversation_chain(k: int = 5):
    retriever = get_retriever(k=k)
    return SimpleRAGConversationChain(retriever)
