Morocco Smart Guide

Objectif
--------
Assistant IA pour générer des itinéraires de voyage personnalisés au Maroc et répondre aux questions des voyageurs.

Fonctionnalités MVP
------------------
- Base de connaissances RAG avec Pinecone ou FAISS
- Chatbot multilingue (FR/EN/ES/DE)
- Génération d’itinéraires jour par jour
- Frontend Streamlit avec formulaire et affichage d’itinéraire

Fonctionnalités bonus
---------------------
- Multi-agents simplifiés (LangGraph)
- Recherche par image (CLIP)
- Infos temps réel (météo, transports)
- Carte interactive des lieux

Stack technique
---------------
- Backend : FastAPI
- LLM / RAG : LangChain + GPT-4
- Base vectorielle : Pinecone / FAISS
- Frontend : Streamlit
- Gestion clés API : python-dotenv

Installation
------------
1. Cloner le dépôt :
   git clone https://github.com/Renzo0023/morocco_smart_guide.git
2. Créer un environnement Python :
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
3. Installer les dépendances :
   pip install -r requirements.txt

Structure du projet
------------------
morocco_smart_guide/
  app/
    __init__.py
    config.py
    data/
    rag/
    itineraries/
    api/
  notebooks/
  tests/
  requirements.txt
  README.txt

Contribution
------------
Pour collaborer :
- Chaque membre travaille sur sa propre branche
- Créer des Pull Requests pour fusionner sur main
- Toujours pull avant push pour éviter les conflits

Auteurs
-------
- ZONGO Nabonswendé Regis Epiphane
- CISSE Marwane
- LANKOANDE Melwine
- MBAIHORNOM Lionel
