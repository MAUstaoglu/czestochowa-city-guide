"""
Embedding utilities using sentence-transformers.
"""

from sentence_transformers import SentenceTransformer
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_MODEL


class EmbeddingModel:
    """Wrapper for sentence-transformer embedding model."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL}...")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model loaded.")
        return self._model
    
    def embed_text(self, text: str) -> list:
        """Embed a single text string."""
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: list) -> list:
        """Embed multiple texts."""
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


# Global instance
embedding_model = EmbeddingModel()


def get_embeddings(texts: list) -> list:
    """Convenience function to get embeddings for a list of texts."""
    return embedding_model.embed_texts(texts)


def get_embedding(text: str) -> list:
    """Convenience function to get embedding for a single text."""
    return embedding_model.embed_text(text)


if __name__ == "__main__":
    # Test embeddings
    test_texts = [
        "What restaurants are in Częstochowa?",
        "Where can I find a good hotel?",
        "Tell me about Jasna Góra monastery."
    ]
    
    print("Testing embedding model...")
    embeddings = get_embeddings(test_texts)
    
    for text, emb in zip(test_texts, embeddings):
        print(f"Text: {text[:50]}...")
        print(f"Embedding dimension: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
        print()
