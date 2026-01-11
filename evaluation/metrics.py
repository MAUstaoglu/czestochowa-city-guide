"""
Evaluation metrics for the QA system.
Measures factual accuracy, latency, and answer relevance.
"""

import json
import os
import sys
import time
from typing import List, Dict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def keyword_overlap_score(answer: str, expected_keywords: List[str]) -> float:
    """
    Calculate keyword overlap between answer and expected keywords.
    
    Returns a score between 0 and 1 indicating what fraction of 
    expected keywords appear in the answer.
    """
    if not answer or not expected_keywords:
        return 0.0
    
    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matches / len(expected_keywords)


def semantic_similarity_score(answer: str, expected_keywords: List[str]) -> float:
    """
    Calculate semantic similarity using embeddings.
    This provides a more nuanced relevance score.
    """
    try:
        from rag.embeddings import get_embedding
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create a reference text from keywords
        reference = " ".join(expected_keywords)
        
        # Get embeddings
        answer_emb = np.array(get_embedding(answer)).reshape(1, -1)
        ref_emb = np.array(get_embedding(reference)).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(answer_emb, ref_emb)[0][0]
        return float(similarity)
    
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return 0.0


def measure_latency(func, *args, **kwargs) -> tuple:
    """
    Measure the execution time of a function.
    
    Returns:
        Tuple of (result, latency_ms)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    latency = (time.perf_counter() - start) * 1000  # Convert to ms
    return result, latency


def calculate_relevance_score(answer: str, retrieved_docs: List[Dict], 
                              expected_category: str = None) -> float:
    """
    Calculate how relevant the retrieved documents are to the query.
    
    Considers:
    - Whether documents match expected category
    - Document distances (lower is better)
    """
    if not retrieved_docs:
        return 0.0
    
    scores = []
    for doc in retrieved_docs:
        score = 1.0
        
        # Distance penalty (convert distance to similarity)
        distance = doc.get("distance", 1.0)
        score *= (1 - min(distance, 1.0))
        
        # Category bonus
        if expected_category:
            doc_category = doc.get("metadata", {}).get("category", "")
            if doc_category == expected_category:
                score *= 1.2  # 20% bonus for matching category
        
        scores.append(min(score, 1.0))
    
    return np.mean(scores)


def evaluate_response(question: str, answer: str, expected_keywords: List[str],
                     sources: List[Dict] = None, expected_category: str = None,
                     latency_ms: float = None) -> Dict:
    """
    Comprehensive evaluation of a single QA response.
    
    Returns a dictionary with various evaluation metrics.
    """
    metrics = {}
    
    # Keyword overlap
    metrics["keyword_overlap"] = keyword_overlap_score(answer, expected_keywords)
    
    # Semantic similarity
    metrics["semantic_similarity"] = semantic_similarity_score(answer, expected_keywords)
    
    # Answer length (normalized)
    word_count = len(answer.split())
    metrics["answer_length"] = word_count
    metrics["answer_length_score"] = min(word_count / 50, 1.0) if word_count > 10 else word_count / 50
    
    # Relevance of retrieved documents
    if sources:
        metrics["retrieval_relevance"] = calculate_relevance_score(
            answer, sources, expected_category
        )
    
    # Latency
    if latency_ms is not None:
        metrics["latency_ms"] = latency_ms
        # Score based on latency (faster is better, <2s is excellent)
        metrics["latency_score"] = max(0, 1 - (latency_ms / 5000))
    
    # Combined score (weighted average)
    weights = {
        "keyword_overlap": 0.3,
        "semantic_similarity": 0.3,
        "retrieval_relevance": 0.2,
        "latency_score": 0.1,
        "answer_length_score": 0.1
    }
    
    combined = 0
    total_weight = 0
    for metric, weight in weights.items():
        if metric in metrics and metrics[metric] is not None:
            combined += metrics[metric] * weight
            total_weight += weight
    
    metrics["combined_score"] = combined / total_weight if total_weight > 0 else 0
    
    return metrics


class EvaluationReport:
    """Collects and summarizes evaluation results."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, question: str, answer: str, metrics: Dict):
        """Add an evaluation result."""
        self.results.append({
            "question": question,
            "answer": answer[:200] + "..." if len(answer) > 200 else answer,
            "metrics": metrics
        })
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all evaluations."""
        if not self.results:
            return {}
        
        # Aggregate metrics
        metric_names = ["keyword_overlap", "semantic_similarity", "combined_score", 
                       "latency_ms", "retrieval_relevance"]
        
        summary = {
            "total_questions": len(self.results),
            "average_metrics": {},
            "min_metrics": {},
            "max_metrics": {}
        }
        
        for metric in metric_names:
            values = [r["metrics"].get(metric) for r in self.results 
                     if r["metrics"].get(metric) is not None]
            if values:
                summary["average_metrics"][metric] = np.mean(values)
                summary["min_metrics"][metric] = np.min(values)
                summary["max_metrics"][metric] = np.max(values)
        
        return summary
    
    def print_report(self):
        """Print a formatted evaluation report."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print(f"\nTotal Questions Evaluated: {summary.get('total_questions', 0)}")
        
        print("\n--- Average Metrics ---")
        for metric, value in summary.get("average_metrics", {}).items():
            if "latency" in metric:
                print(f"  {metric}: {value:.2f} ms")
            else:
                print(f"  {metric}: {value:.3f}")
        
        print("\n--- Performance Summary ---")
        avg_combined = summary.get("average_metrics", {}).get("combined_score", 0)
        if avg_combined >= 0.7:
            print("  ✅ System performance: GOOD")
        elif avg_combined >= 0.5:
            print("  ⚠️  System performance: ACCEPTABLE")
        else:
            print("  ❌ System performance: NEEDS IMPROVEMENT")
        
        avg_latency = summary.get("average_metrics", {}).get("latency_ms", 0)
        if avg_latency < 2000:
            print(f"  ✅ Response time: FAST ({avg_latency:.0f}ms avg)")
        elif avg_latency < 5000:
            print(f"  ⚠️  Response time: MODERATE ({avg_latency:.0f}ms avg)")
        else:
            print(f"  ❌ Response time: SLOW ({avg_latency:.0f}ms avg)")
        
        print("="*60)
    
    def save_report(self, filepath: str):
        """Save the report to a JSON file."""
        report_data = {
            "summary": self.get_summary(),
            "detailed_results": self.results
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved to {filepath}")
