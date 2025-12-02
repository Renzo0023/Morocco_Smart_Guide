# app/config.py
from pathlib import Path
from dotenv import load_dotenv
import os

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load variables from .env file
load_dotenv(ENV_PATH)

# -------------------------
# Hugging Face configuration (Open-Source Models)
# -------------------------

# If you use HuggingFace Inference API (OPTIONAL)
HF_API_KEY = os.getenv("HF_API_KEY", None)

# Embedding model (multilingual + léger)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Local LLM for text generation
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "mistralai/Mistral-7B-Instruct-v0.2"  # On peut ajuster si trop lourd
)

# -------------------------
# Data paths
# -------------------------
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index"

# Create dirs if needed
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
