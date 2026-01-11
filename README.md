# Częstochowa City Guide - AI-Powered QA System

A Retrieval-Augmented Generation (RAG) based Question-Answering system for the city of Częstochowa, Poland. Built as a Neural Networks course project demonstrating modern deep learning techniques.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🏰 Overview

This project implements an intelligent city guide chatbot that can answer questions about:
- 🍽️ Restaurants and cafes
- 🏨 Hotels and accommodations  
- ⛪ Religious sites (including famous Jasna Góra Monastery)
- 🏛️ Museums and attractions
- 🌳 Parks and historic sites

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Flask)                     │
├─────────────────────────────────────────────────────────────┤
│                      RAG Pipeline                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Query     │→ │  ChromaDB   │→ │  Gemma:2b (Ollama)  │  │
│  │  Embedding  │  │  Retrieval  │  │     Generation      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│              Vector Database (ChromaDB)                      │
│         Enriched POI Data from OpenStreetMap                 │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** with Gemma:2b model

### Install Ollama

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai

# Pull the Gemma model
ollama pull gemma:2b
```

## 🚀 Quick Start

### 1. Clone and Install Dependencies

```bash
cd project
pip install -r requirements.txt
```

### 2. Fetch and Prepare Data

```bash
# Fetch POIs from OpenStreetMap
python data/fetch_osm_data.py

# Enrich with sample reviews
python data/generate_reviews.py

# Index into vector database
python rag/vector_store.py
```

### 3. Start Ollama

```bash
ollama serve
```

### 4. Run the Application

```bash
python app.py
```

Open your browser at **http://localhost:5000**

## 📁 Project Structure

```
project/
├── app.py                 # Flask web server
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
│
├── data/
│   ├── fetch_osm_data.py  # OpenStreetMap data fetcher
│   ├── generate_reviews.py # Review generator
│   └── czestochowa_pois.json # Enriched POI data
│
├── rag/
│   ├── embeddings.py      # Sentence embeddings
│   ├── vector_store.py    # ChromaDB integration
│   ├── llm.py             # Ollama/Gemma integration
│   └── pipeline.py        # Complete RAG pipeline
│
├── evaluation/
│   ├── metrics.py         # Evaluation metrics
│   ├── test_questions.json # Test dataset
│   └── run_evaluation.py  # Benchmark runner
│
├── templates/
│   └── index.html         # Chat interface
│
└── static/
    └── style.css          # Styling
```

## 📊 Evaluation

Run the evaluation benchmark:

```bash
python evaluation/run_evaluation.py
```

This measures:
- **Keyword Overlap**: Factual accuracy based on expected keywords
- **Semantic Similarity**: Relevance using embedding similarity
- **Latency**: Response time in milliseconds
- **Retrieval Relevance**: Quality of retrieved documents

## 🔧 Configuration

Edit `config.py` to customize:

```python
# LLM settings
OLLAMA_MODEL = "gemma:2b"      # Can use gemma:7b for better quality
TOP_K_RESULTS = 3              # Number of documents to retrieve

# Server settings  
FLASK_PORT = 5000
```

## 💡 Example Questions

- "What restaurants are in Częstochowa?"
- "Tell me about Jasna Góra monastery"
- "Where can I find a good hotel?"
- "What museums can I visit?"
- "Recommend a cafe with good ratings"

## 🎓 Course Information

**Course**: Neural Networks and Machine Learning  
**Project**: Deep Learning-Based QA System  
**Student**: Mehmet Ali Ustaoglu

## 📄 License

MIT License - feel free to use for educational purposes.
