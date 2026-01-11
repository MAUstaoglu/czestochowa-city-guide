"""
Complete RAG (Retrieval-Augmented Generation) pipeline.
Combines vector search with LLM generation for question answering.
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K_RESULTS
from rag.vector_store import VectorStore
from rag.llm import LLM


class RAGPipeline:
    """RAG Pipeline for the Częstochowa City Guide QA System."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        print("Initializing RAG Pipeline...")
        self.vector_store = VectorStore()
        self.llm = LLM()
        
        # Check LLM availability
        self._llm_available = None
    
    def check_llm(self) -> bool:
        """Check if LLM is available."""
        if self._llm_available is None:
            self._llm_available = self.llm.is_available()
        return self._llm_available
    
    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS, 
                 category: str = None) -> list:
        """Retrieve relevant documents for a query."""
        results = self.vector_store.search(query, top_k=top_k, category_filter=category)
        return results
    
    def build_context(self, retrieved_docs: list) -> str:
        """Build context string from retrieved documents."""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Source {i}]: {doc['document']}")
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate an answer using the LLM with the provided context."""
        if not self.check_llm():
            return self._fallback_answer(query, context)
        
        return self.llm.generate(query, context=context)
    
    def _fallback_answer(self, query: str, context: str) -> str:
        """Provide a fallback answer when LLM is not available."""
        if not context:
            return "I couldn't find relevant information to answer your question."
        
        # Simple extractive fallback
        return f"Based on the available information:\n\n{context[:1000]}..."
    
    def query(self, question: str, top_k: int = TOP_K_RESULTS, 
              category: str = None, return_sources: bool = False) -> dict:
        """
        Complete RAG query: retrieve → augment → generate.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            category: Optional category filter
            return_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        retrieved_docs = self.retrieve(question, top_k=top_k, category=category)
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Build context from retrieved documents
        context = self.build_context(retrieved_docs)
        
        # Step 3: Generate answer
        generation_start = time.time()
        answer = self.generate_answer(question, context)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Build response
        response = {
            "answer": answer,
            "metadata": {
                "total_time_ms": round(total_time * 1000, 2),
                "retrieval_time_ms": round(retrieval_time * 1000, 2),
                "generation_time_ms": round(generation_time * 1000, 2),
                "documents_retrieved": len(retrieved_docs),
                "llm_available": self.check_llm()
            }
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "name": doc["metadata"].get("name", "Unknown"),
                    "category": doc["metadata"].get("category", "N/A"),
                    "rating": doc["metadata"].get("rating", 0),
                    "relevance_score": 1 - doc["distance"]  # Convert distance to similarity
                }
                for doc in retrieved_docs
            ]
        
        return response
    
    def query_stream(self, question: str, top_k: int = TOP_K_RESULTS,
                    category: str = None):
        """
        Streaming RAG query that yields response chunks.
        
        Yields tuples of (chunk_type, content) where chunk_type is:
        - 'sources': Retrieved source documents
        - 'answer': Answer text chunk
        - 'done': Final metadata
        """
        # Retrieve documents
        retrieved_docs = self.retrieve(question, top_k=top_k, category=category)
        
        # Yield sources first
        sources = [
            {
                "name": doc["metadata"].get("name", "Unknown"),
                "category": doc["metadata"].get("category", "N/A"),
                "rating": doc["metadata"].get("rating", 0)
            }
            for doc in retrieved_docs
        ]
        yield ("sources", sources)
        
        # Build context
        context = self.build_context(retrieved_docs)
        
        # Generate streaming answer
        if self.check_llm():
            for chunk in self.llm.generate_stream(question, context=context):
                yield ("answer", chunk)
        else:
            yield ("answer", self._fallback_answer(question, context))
        
        yield ("done", {"documents_retrieved": len(retrieved_docs)})
    
    def get_categories(self) -> list:
        """Get all available POI categories."""
        return self.vector_store.get_all_categories()


# Create global pipeline instance
_pipeline = None

def get_pipeline() -> RAGPipeline:
    """Get or create the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    
    pipeline = get_pipeline()
    
    # Check if documents are indexed
    doc_count = pipeline.vector_store.collection.count()
    if doc_count == 0:
        print("\n⚠️  No documents indexed. Please run:")
        print("  1. python data/fetch_osm_data.py")
        print("  2. python data/generate_reviews.py")
        print("  3. python rag/vector_store.py")
        exit(1)
    
    print(f"\n✅ Found {doc_count} indexed documents")
    print(f"✅ LLM available: {pipeline.check_llm()}")
    
    # Test queries
    test_questions = [
        "What are the best restaurants in Częstochowa?",
        "Where can I visit religious sites?",
        "Recommend a hotel with good ratings."
    ]
    
    print("\n--- Running Test Queries ---")
    for question in test_questions:
        print(f"\n📝 Question: {question}")
        result = pipeline.query(question, return_sources=True)
        print(f"💬 Answer: {result['answer'][:300]}...")
        print(f"⏱️  Time: {result['metadata']['total_time_ms']}ms")
        if result.get("sources"):
            print("📚 Sources:")
            for src in result["sources"][:2]:
                print(f"   - {src['name']} ({src['category']})")
