# app/rag/qa_chain.py

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.rag.vectorstore import get_retriever


def get_llm():
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0.4,
        openai_api_key=OPENAI_API_KEY
    )


# Nouveau : ConversationalRetrievalChain avec mémoire
def get_rag_conversation_chain(k: int = 5):
    retriever = get_retriever(k=k)
    llm = get_llm()

    # Mémoire : garde tout l'historique de la session
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Prompt personnalisé
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Tu es un expert du tourisme au Maroc (toutes villes : Marrakech, Fès, Tanger, Agadir, Rabat, etc.)\n"
            "Réponds uniquement en utilisant le contexte fourni.\n\n"
            "=== CONTEXTE ===\n{context}\n"
            "=== QUESTION ===\n{question}\n\n"
            "Réponds dans un style clair, humain et utile. Si l'information n'est pas dans le contexte, dis-le."
        )
    )

    # Chaîne RAG + mémoire
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return chain

