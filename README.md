**Morocco Smart Guide


*Morocco Smart Guide* est un assistant touristique intelligent basé sur l’intelligence artificielle, conçu pour faciliter la planification de voyages au Maroc à travers des itinéraires personnalisés et une assistance conversationnelle contextuelle.

Le projet repose sur une architecture moderne, modulaire et entièrement open-source, combinant recherche sémantique, génération contrôlée de texte et planification algorithmique.


🎯 Objectif du projet

L’objectif principal du projet est de proposer une application capable de :

* Générer des itinéraires touristiques personnalisés et structurés au Maroc
* Fournir une assistance conversationnelle via un chatbot IA contextuel
* Exploiter une base de connaissances touristique dédiée à l’aide d’un système RAG
* S’adapter aux préférences des utilisateurs : budget, centres d’intérêt, contraintes et durée du séjour

Contrairement à un simple usage d’un modèle de langage généraliste, la solution proposée repose sur une intégration contrôlée de l’IA, garantissant des résultats cohérents, fiables et directement exploitables.


🚀 Fonctionnalités — MVP

1. Base de connaissances touristique (RAG)

* Données touristiques structurées sous forme de fichiers CSV multi-villes
* Chargement et normalisation automatique des données
* Index vectoriel FAISS local persisté sur disque
* Embeddings multilingues basés sur le modèle
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2


2. Chatbot touristique IA (FR / EN)

* Moteur conversationnel basé sur une chaîne RAG personnalisée
* Recherche sémantique sur la base FAISS
* Mémoire de session conservée côté backend
* Réponses contextualisées ancrées dans la base de connaissances
* Interaction en langage naturel avec continuité conversationnelle


3. Génération d’itinéraires jour par jour

* Sélection des lieux via recherche sémantique (FAISS)
* Planification déterministe des activités par jour et créneau
  (matin / après-midi / soir)
* Prise en compte explicite des paramètres :

  * budget
  * centres d’intérêt
  * contraintes
  * durée du séjour
* Génération finale structurée en JSON
* Enrichissement contrôlé des descriptions via un LLM Hugging Face
* Modèle recommandé :
  mistralai/Mistral-7B-Instruct-v0.2


4. Frontend Streamlit

* Interface simple et intuitive
* Formulaire complet de génération d’itinéraire
* Visualisation ergonomique des plannings journaliers
* Intégration d’un chatbot avec mémoire de session
* Liens Google Maps pour chaque activité


⭐ Fonctionnalités optionnelles (perspectives)

* Orchestration multi-agents (LangGraph)
* Recherche visuelle par image (CLIP)
* Carte interactive (Folium / Leaflet)
* Intégration d’APIs externes (météo, transports)


🧱 Stack technique

 Backend

* FastAPI
* LangChain (documents, vector stores, RAG)
* FAISS-cpu (base vectorielle locale)

 IA / LLM (open-source)

* Hugging Face Inference API
* Modèle de génération :
  mistralai/Mistral-7B-Instruct-v0.2
* Modèle d’embeddings :
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

 Frontend

* Streamlit

 Outils complémentaires

* python-dotenv (gestion des variables d’environnement)
* httpx (client HTTP)
* jupyter (tests et exploration)

---

 📁 Structure du projet


morocco_smart_guide/
├─ app/
│  ├─ config.py
│  ├─ data/
│  │  └─ loader.py                # CSV → Places → Documents
│  ├─ rag/
│  │  ├─ embeddings.py            # Embeddings Hugging Face
│  │  ├─ vectorstore.py           # FAISS build/load/retriever
│  │  └─ qa_chain.py              # RAG + mémoire (chatbot)
│  ├─ itineraries/
│  │  ├─ models.py                # TravelProfile, Itinerary…
│  │  └─ generator.py             # Génération d’itinéraires
│  └─ api/
│     ├─ schemas.py               # ChatRequest / ChatResponse
│     └─ main.py                  # API FastAPI
│
├─ scripts/
│  ├─ build_faiss_index.py        # Construction index FAISS
│  └─ demo_generate_itinerary.py
│
├─ data/                          # CSV multi-villes
├─ notebooks/                     # Expérimentations
├─ requirements.txt
├─ README.md
└─ .env.example



 🛠️ Installation et lancement

 1. Cloner le dépôt

bash
git clone https://github.com/Renzo0023/morocco_smart_guide.git
cd morocco_smart_guide


 2. Créer un environnement virtuel

bash
python -m venv venv             # Python 3.11 (3.11.9 par exemple)
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows


 3. Installer les dépendances

bash
* pip install -r requirements.txt
* pip install langchain-community
* pip install langchain-community sequence transformers
* pip install langchain-huggingface


 4. Configurer les variables d’environnement

Créer un fichier .env :

env
HF_API_KEY=ton_token_huggingface
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
DATA_DIR=./data
FAISS_INDEX_PATH=./app/rag/faiss_index


 5. Construire l’index vectoriel FAISS

bash
python -m scripts.build_faiss_index


 6. Lancer l’API FastAPI

bash
uvicorn app.api.main:app --reload


 7. Lancer l’interface Streamlit

bash
streamlit run app/ui/app.py


 👥 Auteurs

* ZONGO Nabonswendé Regis Epiphane
* CISSE Marwane
* LANKOANDE Melwine
* MBAIHORNOM Lionel
