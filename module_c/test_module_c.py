"""
Module C - Test Script
=======================
This script tests all retrieval methods on the actual dataset.
"""

import json
import sys
import os

# Add parent directory to path so we can import module_c
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


def test_bm25(documents):
    """Test BM25 Search"""
    print("\n" + "="*60)
    print("🔍 Testing BM25 Search")
    print("="*60)
    
    from module_c.bm25 import BM25Search
    
    bm25 = BM25Search(documents)
    
    # Test queries
    test_queries = [
        "নির্বাচন",  # Bangla: election
        "Bangladesh cricket",  # English
        "বিএনপি",  # Bangla: BNP
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        results = bm25.search(query, k=3)
        
        if results:
            for i, (doc_id, score, doc) in enumerate(results, 1):
                title = doc.get('title', 'N/A')[:50]
                print(f"   {i}. Score: {score:.2f} | {title}...")
        else:
            print("   ❌ No results found")
    
    return True


def test_fuzzy(documents):
    """Test Fuzzy Search"""
    print("\n" + "="*60)
    print("🔍 Testing Fuzzy Search")
    print("="*60)
    
    from module_c.fuzzy import FuzzySearch
    
    fuzzy = FuzzySearch(documents, threshold=60)
    
    # Test queries with typos/variations
    test_queries = [
        "Dhaka",  # Should match ঢাকা related
        "Bangaldesh",  # Typo for Bangladesh
        "electon",  # Typo for election
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}' (fuzzy)")
        results = fuzzy.search(query, k=3)
        
        if results:
            for i, (doc_id, score, doc) in enumerate(results, 1):
                title = doc.get('title', 'N/A')[:50]
                print(f"   {i}. Score: {score:.2f} | {title}...")
        else:
            print("   ❌ No results found")
    
    return True


def test_semantic(documents):
    """Test Semantic Search"""
    print("\n" + "="*60)
    print("🔍 Testing Semantic Search")
    print("="*60)
    
    from module_c.semantic import SemanticSearch
    
    # Use smaller subset for faster testing
    test_docs = documents[:50]
    semantic = SemanticSearch(test_docs)
    
    # Test cross-lingual queries
    test_queries = [
        "election voting",  # English → should find Bangla election articles
        "cricket match",  # English sports
        "রাজধানী",  # Bangla: capital → should find Dhaka articles
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}' (semantic)")
        results = semantic.search(query, k=3)
        
        if results:
            for i, (doc_id, score, doc) in enumerate(results, 1):
                title = doc.get('title', 'N/A')[:50]
                print(f"   {i}. Score: {score:.3f} | {title}...")
        else:
            print("   ❌ No results found")
    
    return True


def test_hybrid(documents):
    """Test Hybrid Search"""
    print("\n" + "="*60)
    print("🔍 Testing Hybrid Search")
    print("="*60)
    
    from module_c.hybrid import HybridSearch
    
    # Use smaller subset for faster testing
    test_docs = documents[:50]
    hybrid = HybridSearch(test_docs, alpha=0.3, beta=0.5, gamma=0.2)
    
    # Test query
    query = "Bangladesh election news"
    print(f"\n📝 Query: '{query}' (hybrid)")
    
    results = hybrid.search(query, k=5, verbose=True)
    
    return True


def main():
    print("="*60)
    print("Module C - Retrieval Models Test Suite")
    print("="*60)
    
    # Path to dataset
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dataset', 'articles_all.jsonl'
    )
    
    print(f"\n📂 Loading documents from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    # Load a subset of documents for testing
    documents = load_documents(dataset_path, limit=100)
    print(f"✅ Loaded {len(documents)} documents")
    
    # Test each method
    try:
        print("\n" + "="*60)
        print("TEST 1: BM25 (Lexical Search)")
        print("="*60)
        test_bm25(documents)
        print("\n✅ BM25 test passed!")
    except Exception as e:
        print(f"\n❌ BM25 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n" + "="*60)
        print("TEST 2: Fuzzy Search")
        print("="*60)
        test_fuzzy(documents)
        print("\n✅ Fuzzy test passed!")
    except Exception as e:
        print(f"\n❌ Fuzzy test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n" + "="*60)
        print("TEST 3: Semantic Search")
        print("="*60)
        test_semantic(documents)
        print("\n✅ Semantic test passed!")
    except Exception as e:
        print(f"\n❌ Semantic test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("\n" + "="*60)
        print("TEST 4: Hybrid Search")
        print("="*60)
        test_hybrid(documents)
        print("\n✅ Hybrid test passed!")
    except Exception as e:
        print(f"\n❌ Hybrid test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("🎉 All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
