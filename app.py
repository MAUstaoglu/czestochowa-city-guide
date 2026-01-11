"""
Flask web application for the Częstochowa City Guide QA System.
Provides a chat interface for interacting with the RAG pipeline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import json

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from rag.pipeline import get_pipeline

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize pipeline (lazy loading)
_pipeline = None

def get_rag_pipeline():
    """Get or initialize the RAG pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = get_pipeline()
    return _pipeline


@app.route("/")
def index():
    """Render the chat interface."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages and return RAG responses."""
    try:
        data = request.get_json()
        question = data.get("message", "").strip()
        
        if not question:
            return jsonify({"error": "No message provided"}), 400
        
        # Get optional parameters
        category = data.get("category")
        include_sources = data.get("include_sources", True)
        
        # Query the RAG pipeline
        pipeline = get_rag_pipeline()
        result = pipeline.query(
            question, 
            category=category,
            return_sources=include_sources
        )
        
        return jsonify({
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "metadata": result["metadata"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Handle chat with streaming response."""
    try:
        data = request.get_json()
        question = data.get("message", "").strip()
        
        if not question:
            return jsonify({"error": "No message provided"}), 400
        
        category = data.get("category")
        pipeline = get_rag_pipeline()
        
        def generate():
            for chunk_type, content in pipeline.query_stream(question, category=category):
                yield f"data: {json.dumps({'type': chunk_type, 'content': content})}\n\n"
        
        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Get available POI categories."""
    try:
        pipeline = get_rag_pipeline()
        categories = pipeline.get_categories()
        return jsonify({"categories": categories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get system status."""
    try:
        pipeline = get_rag_pipeline()
        doc_count = pipeline.vector_store.collection.count()
        llm_available = pipeline.check_llm()
        current_model = pipeline.llm.get_current_model()
        
        return jsonify({
            "status": "ready" if doc_count > 0 else "no_data",
            "documents_indexed": doc_count,
            "llm_available": llm_available,
            "current_model": current_model,
            "categories": pipeline.get_categories()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route("/api/models", methods=["GET"])
def get_models():
    """Get available Ollama models."""
    try:
        pipeline = get_rag_pipeline()
        models = pipeline.llm.get_available_models()
        current = pipeline.llm.get_current_model()
        return jsonify({
            "models": models,
            "current_model": current
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/switch", methods=["POST"])
def switch_model():
    """Switch to a different model."""
    try:
        data = request.get_json()
        model_name = data.get("model", "").strip()
        
        if not model_name:
            return jsonify({"error": "No model specified"}), 400
        
        pipeline = get_rag_pipeline()
        success = pipeline.llm.set_model(model_name)
        
        if success:
            return jsonify({
                "success": True,
                "current_model": pipeline.llm.get_current_model()
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Model '{model_name}' not available"
            }), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Częstochowa City Guide - QA System")
    print("="*60)
    
    # Check system status
    try:
        pipeline = get_rag_pipeline()
        doc_count = pipeline.vector_store.collection.count()
        
        if doc_count == 0:
            print("\n⚠️  Warning: No documents indexed!")
            print("Run the following commands first:")
            print("  1. python data/fetch_osm_data.py")
            print("  2. python data/generate_reviews.py")
            print("  3. python rag/vector_store.py")
        else:
            print(f"\n✅ {doc_count} documents indexed")
        
        if pipeline.check_llm():
            print("✅ LLM (Gemma:2b) is available")
        else:
            print("⚠️  LLM not available - using fallback mode")
            print("   Start Ollama and run: ollama pull gemma:2b")
    
    except Exception as e:
        print(f"\n❌ Error initializing: {e}")
    
    print(f"\n🚀 Starting server at http://{FLASK_HOST}:{FLASK_PORT}")
    print("="*60 + "\n")
    
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
