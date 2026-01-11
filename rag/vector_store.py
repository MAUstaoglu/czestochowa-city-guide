"""
Vector store implementation using ChromaDB.
Handles document indexing and semantic search.
"""

import json
import os
import sys

import chromadb
from chromadb.config import Settings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_DB_DIR, DATA_DIR, TOP_K_RESULTS
from rag.embeddings import get_embedding, get_embeddings


class VectorStore:
    """ChromaDB-based vector store for POI documents."""
    
    def __init__(self, collection_name: str = "czestochowa_pois"):
        """Initialize the vector store."""
        self.collection_name = collection_name
        
        # Initialize ChromaDB with persistence
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Częstochowa Points of Interest"}
        )
    
    def index_documents(self, pois: list, force_reindex: bool = False):
        """Index POI documents into the vector store."""
        # Check if already indexed
        if self.collection.count() > 0 and not force_reindex:
            print(f"Collection already has {self.collection.count()} documents.")
            print("Use force_reindex=True to re-index.")
            return
        
        if force_reindex and self.collection.count() > 0:
            # Delete existing collection and recreate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Częstochowa Points of Interest"}
            )
        
        print(f"Indexing {len(pois)} documents...")
        
        # Prepare data for batch insertion
        ids = []
        documents = []
        metadatas = []
        
        for poi in pois:
            doc_text = poi.get("document_text", "")
            if not doc_text:
                continue
            
            ids.append(str(poi["id"]))
            documents.append(doc_text)
            metadatas.append({
                "name": poi["name"],
                "category": poi["category"],
                "lat": poi["lat"],
                "lon": poi["lon"],
                "rating": poi.get("review_data", {}).get("average_rating", 0)
            })
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = get_embeddings(documents)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            print(f"Indexed {end_idx}/{len(ids)} documents")
        
        print(f"Indexing complete. Total documents: {self.collection.count()}")
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS, category_filter: str = None) -> list:
        """Search for relevant documents based on a query."""
        # Generate query embedding
        query_embedding = get_embedding(query)
        
        # Build where clause for filtering
        where_clause = None
        if category_filter:
            where_clause = {"category": category_filter}
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                    "id": results["ids"][0][i] if results["ids"] else None
                })
        
        return formatted_results
    
    def get_all_categories(self) -> list:
        """Get all unique categories in the collection."""
        # Get a sample of documents to extract categories
        results = self.collection.get(limit=1000, include=["metadatas"])
        categories = set()
        for metadata in results.get("metadatas", []):
            if metadata and "category" in metadata:
                categories.add(metadata["category"])
        return sorted(list(categories))


def load_and_index_pois(force_reindex: bool = False):
    """Load POIs from JSON and index them."""
    poi_file = os.path.join(DATA_DIR, "czestochowa_pois.json")
    
    try:
        with open(poi_file, "r", encoding="utf-8") as f:
            pois = json.load(f)
    except FileNotFoundError:
        print(f"Error: {poi_file} not found.")
        print("Please run data/fetch_osm_data.py and data/generate_reviews.py first.")
        return None
    
    store = VectorStore()
    store.index_documents(pois, force_reindex=force_reindex)
    return store


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index POI documents into ChromaDB")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing")
    args = parser.parse_args()
    
    store = load_and_index_pois(force_reindex=args.reindex)
    
    if store:
        # Test search
        print("\n--- Testing Search ---")
        test_queries = [
            "Where can I eat pierogi?",
            "Best hotels in Częstochowa",
            "Tell me about religious sites"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = store.search(query, top_k=2)
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['metadata'].get('name', 'Unknown')} "
                      f"({result['metadata'].get('category', 'N/A')}) - "
                      f"Distance: {result['distance']:.4f}")
