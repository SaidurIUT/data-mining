"""
Module D - Complete Test Script
================================
Tests all Module D components:
1. Ranking & Scoring with normalized scores
2. Query execution timing
3. Evaluation metrics (Precision, Recall, nDCG, MRR)
4. Error analysis
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_documents(filepath, limit=100):
    """Load documents from JSONL file"""
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            documents.append(doc)
    return documents


def test_ranking_system(documents):
    """Test the ranking system with normalized scores and timing"""
    print("\n" + "=" * 70)
    print("🔍 TEST 1: RANKING SYSTEM")
    print("=" * 70)
    
    from module_d.ranking import RankingSystem, RetrievalMethod
    
    # Initialize ranking system
    ranker = RankingSystem(documents, confidence_threshold=0.20)
    
    # Test queries
    test_queries = [
        ("নির্বাচন", "Bangla: election"),
        ("Bangladesh election", "English query"),
        ("বিএনপি আওয়ামী লীগ", "Bangla: political parties"),
        ("xyz123nonsense", "Should trigger low confidence"),
    ]
    
    for query, description in test_queries:
        print(f"\n📝 Testing: {description}")
        result = ranker.search(query, method=RetrievalMethod.HYBRID, k=5, verbose=True)
    
    # Compare all methods
    print("\n" + "=" * 70)
    print("📊 COMPARING ALL RETRIEVAL METHODS")
    print("=" * 70)
    
    ranker.search_all_methods("নির্বাচন ভোট", k=5)
    
    return ranker


def test_evaluation_metrics(documents, ranker):
    """Test evaluation metrics with PROPER relevance judgments"""
    print("\n" + "=" * 70)
    print("📊 TEST 2: EVALUATION METRICS")
    print("=" * 70)
    
    from module_d.evaluation import Evaluator, RelevanceLabeler
    from module_d.ranking import RetrievalMethod
    
    # =========================================================================
    # PROPER EVALUATION: Using independent relevance judgments
    # =========================================================================
    # 
    # The WRONG way (what we did before):
    #   1. Search for documents
    #   2. Label top results as "relevant"  ← CIRCULAR! Results are biased!
    #
    # The RIGHT way (what we do now):
    #   1. Pool documents from multiple methods
    #   2. SHUFFLE them (hide rankings)
    #   3. Human judges relevance INDEPENDENTLY of rank
    #   4. THEN evaluate against these judgments
    # =========================================================================
    
    judgment_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "proper_relevance_judgments.csv"
    )
    
    # Check if proper judgments exist
    if os.path.exists(judgment_file):
        print(f"\n✅ Found existing relevance judgments: {judgment_file}")
        labeler = RelevanceLabeler(judgment_file)
    else:
        print("\n⚠️  No proper relevance judgments found!")
        print("   Creating SIMULATED independent judgments for demo...")
        print("   (In real evaluation, use create_relevance_judgments.py)")
        
        # Simulate proper judgments by:
        # 1. Pooling from multiple methods
        # 2. Labeling based on CONTENT matching (not ranking)
        labeler = create_simulated_proper_judgments(documents, judgment_file)
    
    # Initialize evaluator
    evaluator = Evaluator(labeler)
    
    # Evaluate system
    print("\n📊 Running system evaluation...")
    evaluation = evaluator.evaluate_system(
        ranker,
        queries=labeler.get_all_queries(),
        method=RetrievalMethod.HYBRID,
        k=50,
        verbose=True
    )
    
    return evaluator


def create_simulated_proper_judgments(documents, output_file):
    """
    Create simulated but PROPER relevance judgments.
    
    Instead of labeling top-ranked docs as relevant, we:
    1. Define what makes a document relevant (keyword presence)
    2. Label ALL documents in pool based on this criteria
    3. This simulates independent human judgment
    """
    from module_d.evaluation import RelevanceLabeler
    from module_d.ranking import RankingSystem, RetrievalMethod
    import random
    
    print("\n📋 Creating simulated proper relevance judgments...")
    
    # Relevance criteria: document must contain these terms to be "relevant"
    # This simulates a human judging based on content, not ranking
    relevance_keywords = {
        "নির্বাচন": ["নির্বাচন", "ভোট", "প্রার্থী", "ইসি", "কমিশন", "election", "vote"],
        "বিএনপি": ["বিএনপি", "বিএনপিতে", "তারেক", "খালেদা", "জিয়া", "BNP"],
        "ঢাকা": ["ঢাকা", "রাজধানী", "Dhaka", "মহানগরী"],
    }
    
    ranker = RankingSystem(documents, confidence_threshold=0.20)
    labeler = RelevanceLabeler(output_file)
    
    for query, keywords in relevance_keywords.items():
        print(f"\n   Query: '{query}'")
        
        # Pool documents from multiple methods
        pooled_doc_ids = set()
        for method in [RetrievalMethod.BM25, RetrievalMethod.SEMANTIC, RetrievalMethod.HYBRID]:
            try:
                result = ranker.search(query, method=method, k=20, verbose=False)
                pooled_doc_ids.update(r.doc_id for r in result.results)
            except:
                pass
        
        # Also add some random documents (to test that system doesn't retrieve irrelevant ones)
        all_doc_ids = list(range(len(documents)))
        random_ids = random.sample([d for d in all_doc_ids if d not in pooled_doc_ids], 
                                   min(10, len(all_doc_ids) - len(pooled_doc_ids)))
        pooled_doc_ids.update(random_ids)
        
        print(f"      Pooled {len(pooled_doc_ids)} documents")
        
        # Label based on CONTENT, not ranking
        relevant_count = 0
        for doc_id in pooled_doc_ids:
            doc = documents[doc_id]
            text = (doc.get('title', '') + ' ' + doc.get('body', '')).lower()
            
            # Check if any keyword is present
            is_relevant = any(kw.lower() in text for kw in keywords)
            
            labeler.add_judgment(
                query=query,
                doc_url=doc.get('url', f'doc_{doc_id}'),
                doc_id=doc_id,
                language=doc.get('language', 'bn'),
                relevant=is_relevant,
                annotator="content_based_simulation"
            )
            
            if is_relevant:
                relevant_count += 1
        
        print(f"      Labeled: {relevant_count} relevant, {len(pooled_doc_ids) - relevant_count} not relevant")
    
    labeler.save()
    print(f"\n💾 Saved to {output_file}")
    
    return labeler


def test_error_analysis(documents, ranker):
    """Test error analysis with specific cases"""
    print("\n" + "=" * 70)
    print("🔬 TEST 3: ERROR ANALYSIS")
    print("=" * 70)
    
    from module_d.error_analysis import ErrorAnalyzer
    
    # Initialize error analyzer
    analyzer = ErrorAnalyzer(
        ranking_system=ranker,
        output_dir=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "error_analysis_output"
        )
    )
    
    # Run comprehensive analysis
    case_studies = analyzer.run_comprehensive_analysis(verbose=True)
    
    # Generate reports
    print("\n📄 Generating reports...")
    analyzer.generate_report()
    analyzer.save_case_studies_json()
    
    return analyzer


def test_search_engine_comparison(documents, ranker):
    """Generate template for search engine comparison"""
    print("\n" + "=" * 70)
    print("🌐 TEST 4: SEARCH ENGINE COMPARISON TEMPLATE")
    print("=" * 70)
    
    from module_d.evaluation import SearchEngineComparator
    
    comparator = SearchEngineComparator(
        output_dir=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "comparisons"
        )
    )
    
    # Generate comparison template
    test_queries = [
        "নির্বাচন",
        "Bangladesh election",
        "বিএনপি আওয়ামী লীগ",
        "ঢাকা বিশ্ববিদ্যালয়",
        "cricket Bangladesh",
    ]
    
    comparator.generate_comparison_template(test_queries)
    
    print("\n✅ Comparison template generated!")
    print("   Fill in the template manually by searching on Google, Bing, DuckDuckGo, etc.")
    
    return comparator


def main():
    print("=" * 70)
    print("🧪 MODULE D - COMPLETE TEST SUITE")
    print("=" * 70)
    
    # Path to dataset
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dataset', 'articles_all.jsonl'
    )
    
    print(f"\n📂 Loading documents from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    # Load documents
    documents = load_documents(dataset_path, limit=100)
    print(f"✅ Loaded {len(documents)} documents\n")
    
    # Test 1: Ranking System
    ranker = test_ranking_system(documents)
    
    # Test 2: Evaluation Metrics
    evaluator = test_evaluation_metrics(documents, ranker)
    
    # Test 3: Error Analysis
    analyzer = test_error_analysis(documents, ranker)
    
    # Test 4: Search Engine Comparison Template
    comparator = test_search_engine_comparison(documents, ranker)
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 ALL MODULE D TESTS COMPLETED!")
    print("=" * 70)
    
    print("\n📁 Generated Files:")
    print("   - test_relevance_judgments.csv (sample relevance labels)")
    print("   - error_analysis_output/error_analysis_report.md")
    print("   - error_analysis_output/case_studies.json")
    print("   - comparisons/comparison_template.md")
    
    print("\n📋 Module D Summary:")
    print("   ✅ Ranking & Scoring: Normalized scores (0-1), low-confidence warnings")
    print("   ✅ Query Timing: Breakdown of preprocessing, embedding, ranking time")
    print("   ✅ Evaluation: Precision@K, Recall@K, nDCG@K, MRR")
    print("   ✅ Error Analysis: 5 categories with detailed case studies")
    print("   ✅ Search Engine Comparison: Template for manual comparison")


if __name__ == "__main__":
    main()
