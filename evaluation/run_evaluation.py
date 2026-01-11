"""
Run comprehensive evaluation of the QA system.
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import evaluate_response, measure_latency, EvaluationReport
from rag.pipeline import get_pipeline


def load_test_questions(filepath: str = None) -> list:
    """Load test questions from JSON file."""
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), 
            "test_questions.json"
        )
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(verbose: bool = True):
    """Run full evaluation on the test question set."""
    
    # Load test questions
    test_questions = load_test_questions()
    print(f"Loaded {len(test_questions)} test questions")
    
    # Initialize pipeline
    print("Initializing RAG pipeline...")
    pipeline = get_pipeline()
    
    # Check system readiness
    doc_count = pipeline.vector_store.collection.count()
    if doc_count == 0:
        print("\n❌ Error: No documents indexed!")
        print("Please run the data pipeline first:")
        print("  1. python data/fetch_osm_data.py")
        print("  2. python data/generate_reviews.py")
        print("  3. python rag/vector_store.py")
        return None
    
    print(f"✅ Found {doc_count} indexed documents")
    print(f"✅ LLM available: {pipeline.check_llm()}")
    
    # Run evaluation
    report = EvaluationReport()
    
    print("\nRunning evaluation...")
    for test in tqdm(test_questions, desc="Evaluating"):
        question = test["question"]
        expected_keywords = test.get("expected_keywords", [])
        category_hint = test.get("category_hint")
        
        # Query the pipeline with timing
        result, latency = measure_latency(
            pipeline.query, 
            question, 
            return_sources=True
        )
        
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        
        # Evaluate the response
        metrics = evaluate_response(
            question=question,
            answer=answer,
            expected_keywords=expected_keywords,
            sources=[{"distance": 1-s.get("relevance_score", 0), "metadata": s} 
                    for s in sources],
            expected_category=category_hint,
            latency_ms=latency
        )
        
        report.add_result(question, answer, metrics)
        
        if verbose:
            print(f"\n  Q: {question[:50]}...")
            print(f"  Score: {metrics['combined_score']:.2f} | "
                  f"Latency: {metrics.get('latency_ms', 0):.0f}ms")
    
    # Print and save report
    report.print_report()
    
    # Save report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "evaluation_report.json"
    )
    report.save_report(report_path)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate the QA system")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    run_evaluation(verbose=not args.quiet)
