"""
Configuration settings for the Częstochowa City Guide QA System.
"""

import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

# OpenStreetMap settings
OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CZESTOCHOWA_BBOX = {
    "south": 50.7500,
    "west": 19.0500,
    "north": 50.8500,
    "east": 19.1800
}

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM settings (Ollama)
OLLAMA_MODEL = "gemma:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
FLASK_DEBUG = True
