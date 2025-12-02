🇲🇦 Morocco Smart Guide — README
🎯 Objectif du projet

Morocco Smart Guide est un assistant touristique intelligent capable de :

Générer des itinéraires de voyage personnalisés au Maroc

Répondre aux questions des voyageurs via un chatbot IA multilingue

Fournir des fiches détaillées pour chaque lieu (RAG)

S’adapter aux préférences : budget, centres d’intérêt, contraintes, durée...

Le projet repose sur une architecture moderne, modulaire et 100% open-source
(LLM & embeddings Hugging Face, FAISS, LangChain).

🚀 Fonctionnalités — MVP
✔️ 1. Base de connaissances touristique (RAG)

Données structurées dans des fichiers CSV multi-villes

Index vectoriel FAISS local

Embeddings multilingues :
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

✔️ 2. Chatbot IA (FR/EN)

Moteur conversationnel basé sur LangChain

RAG + mémoire via ConversationalRetrievalChain

Réponses dans la langue de l’utilisateur

✔️ 3. Génération d’itinéraires jour par jour

Utilisation d’un LLM Hugging Face hébergé (ex : Mistral 7B)

Itinéraires structurés (matin / après-midi / soir) en JSON

Prise en compte :

budget

centres d’intérêt

contraintes

multi-villes

✔️ 4. Frontend Streamlit

Formulaire complet : ville(s), durée, budget, intérêts, contraintes

Affichage ergonomique de l’itinéraire

Onglet chatbot avec mémoire de session

⭐ Fonctionnalités optionnelles (si le temps le permet)

Workflow multi-agents (LangGraph)

Recherche par image (CLIP)

Carte interactive (Leaflet / Folium)

Météo / transports (API externes)

🧱 Stack technique
Backend

FastAPI

LangChain (RAG, prompts, orchestration)

FAISS-cpu (vector store)

IA / LLM (Open-source)

HuggingFace Hub (LLM)

Modèle recommandé :
mistralai/Mistral-7B-Instruct-v0.2

Embeddings :
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Frontend

Streamlit

Divers

python-dotenv (gestion .env)

httpx (client HTTP)

jupyter (tests / exploration)

📁 Structure du projet
morocco_smart_guide/
├─ app/
│  ├─ config.py
│  ├─ data/
│  │  ├─ loader.py                # Charge CSV multi-villes → Places → Documents
│  ├─ rag/
│  │  ├─ embeddings.py            # Embeddings Hugging Face
│  │  ├─ vectorstore.py           # FAISS build/load/retriever
│  │  └─ qa_chain.py              # RAG + mémoire (chatbot)
│  ├─ itineraries/
│  │  ├─ models.py                # TravelProfile, Itinerary, etc.
│  │  └─ generator.py             # Génération d’itinéraires via LLM HF
│  └─ api/
│     ├─ schemas.py               # ChatRequest / ChatResponse
│     └─ main.py                  # Endpoints FastAPI
│
├─ scripts/
│  ├─ build_faiss_index.py        # Construction index FAISS
│  └─ demo_generate_itinerary.py  # Tests rapides
│
├─ data/                          # CSV multi-villes (Marrakech, Fès...)
├─ notebooks/                     # Expérimentations
├─ requirements.txt
├─ README.md
└─ .env.example

🛠️ Installation & Lancement
1. Cloner le dépôt
git clone https://github.com/Renzo0023/morocco_smart_guide.git
cd morocco_smart_guide

2. Créer un environnement Python
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

3. Installer les dépendances
pip install -r requirements.txt

4. Configurer les variables d’environnement

Créer un fichier .env :

HF_API_KEY=ton_token_huggingface
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
DATA_DIR=./data
FAISS_INDEX_PATH=./app/rag/faiss_index

5. Construire l’index vectoriel FAISS
python -m scripts.build_faiss_index

6. Lancer l’API FastAPI
uvicorn app.api.main:app --reload

7. Lancer l’interface Streamlit
streamlit run app/ui/app.py    # si tu as un fichier Streamlit

🧪 Exemple d’appel API
Génération d’itinéraire
curl -X POST "http://localhost:8000/itinerary" \
-H "Content-Type: application/json" \
-d '{
  "city": "Marrakech",
  "duration_days": 3,
  "budget": "medium",
  "interests": ["culture", "gastronomy"],
  "constraints": "éviter trop de marche",
  "language": "fr"
}'

Chatbot
curl -X POST "http://localhost:8000/chat" \
-H "Content-Type: application/json" \
-d '{"message": "Que visiter à Marrakech en 2 jours ?"}'

🤝 Contribution

Chaque membre travaille sur sa propre branche

Pull avant push

Pull requests sur main pour fusion

Code documenté + tests minimaux

👥 Auteurs

ZONGO Nabonswendé Regis Epiphane

CISSE Marwane

LANKOANDE Melwine

MBAIHORNOM Lionel