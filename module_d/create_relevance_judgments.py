"""
Proper Relevance Judgment Creation
===================================
This script creates relevance judgments using POOLING method:
1. Run multiple retrieval methods for each query
2. Combine (pool) unique documents from all methods
3. Present documents in RANDOM order (hiding rankings)
4. Human labels each document as relevant/not relevant

This avoids the circular evaluation problem where we just label
top-ranked documents as "relevant".
"""

import json
import os
import sys
import random
from typing import List, Dict, Set
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_documents(filepath: str, limit: int = None) -> List[Dict]:
    """Load documents from JSONL file"""
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            doc['_id'] = i  # Add ID for reference
            documents.append(doc)
    return documents


def create_document_pool(
    documents: List[Dict],
    queries: List[str],
    pool_depth: int = 20
) -> Dict[str, List[Dict]]:
    """
    Create a pool of documents for each query using multiple retrieval methods.
    Documents are shuffled to avoid position bias during labeling.
    
    Args:
        documents: All documents
        queries: List of queries to evaluate
        pool_depth: How many docs to take from each method
    
    Returns:
        Dict mapping query -> list of pooled documents (shuffled)
    """
    from module_d.ranking import RankingSystem, RetrievalMethod
    
    print("=" * 70)
    print("📊 CREATING DOCUMENT POOLS FOR EVALUATION")
    print("=" * 70)
    
    # Initialize ranking system
    ranker = RankingSystem(documents, confidence_threshold=0.20)
    
    query_pools = {}
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Collect documents from each method
        seen_doc_ids: Set[int] = set()
        pooled_docs = []
        
        for method in [RetrievalMethod.BM25, RetrievalMethod.SEMANTIC, RetrievalMethod.HYBRID]:
            try:
                result = ranker.search(query, method=method, k=pool_depth, verbose=False)
                
                for r in result.results:
                    if r.doc_id not in seen_doc_ids:
                        seen_doc_ids.add(r.doc_id)
                        pooled_docs.append({
                            'doc_id': r.doc_id,
                            'title': r.document.get('title', 'N/A'),
                            'body': r.document.get('body', '')[:500],  # First 500 chars
                            'url': r.document.get('url', 'N/A'),
                            'language': r.document.get('language', 'N/A'),
                            # Store which methods retrieved this doc (for analysis later)
                            '_methods': [method.value]
                        })
                    else:
                        # Document already in pool, add this method to its list
                        for doc in pooled_docs:
                            if doc['doc_id'] == r.doc_id:
                                if method.value not in doc['_methods']:
                                    doc['_methods'].append(method.value)
                                break
                
                print(f"   {method.value}: retrieved {len(result.results)} docs, pool now has {len(pooled_docs)} unique")
            
            except Exception as e:
                print(f"   {method.value}: failed - {e}")
        
        # IMPORTANT: Shuffle to remove position bias during labeling
        random.shuffle(pooled_docs)
        query_pools[query] = pooled_docs
        
        print(f"   ✅ Final pool size: {len(pooled_docs)} documents (shuffled)")
    
    return query_pools


def interactive_labeling(query_pools: Dict[str, List[Dict]], output_file: str):
    """
    Interactive terminal-based labeling interface.
    Shows documents one by one and asks for relevance judgment.
    """
    print("\n" + "=" * 70)
    print("📝 INTERACTIVE RELEVANCE LABELING")
    print("=" * 70)
    print("\nInstructions:")
    print("  - You'll see documents one by one for each query")
    print("  - Judge if the document is RELEVANT to the query")
    print("  - Enter: y = relevant, n = not relevant, s = skip, q = quit query")
    print("  - Judgments are saved after each query")
    print("=" * 70)
    
    all_judgments = []
    
    for query, docs in query_pools.items():
        print(f"\n{'='*70}")
        print(f"🔍 QUERY: '{query}'")
        print(f"   {len(docs)} documents to judge")
        print(f"{'='*70}")
        
        input("\nPress Enter to start labeling...")
        
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1}/{len(docs)} ---")
            print(f"Title: {doc['title']}")
            print(f"Language: {doc['language']}")
            print(f"\nBody preview:")
            print(f"{doc['body'][:300]}...")
            print(f"\nURL: {doc['url']}")
            
            while True:
                response = input(f"\nIs this relevant to '{query}'? (y/n/s/q): ").lower().strip()
                
                if response == 'y':
                    all_judgments.append({
                        'query': query,
                        'doc_id': doc['doc_id'],
                        'doc_url': doc['url'],
                        'language': doc['language'],
                        'relevant': 'yes',
                        'annotator': 'human',
                        'notes': f"methods: {','.join(doc['_methods'])}"
                    })
                    print("   ✅ Marked as RELEVANT")
                    break
                elif response == 'n':
                    all_judgments.append({
                        'query': query,
                        'doc_id': doc['doc_id'],
                        'doc_url': doc['url'],
                        'language': doc['language'],
                        'relevant': 'no',
                        'annotator': 'human',
                        'notes': f"methods: {','.join(doc['_methods'])}"
                    })
                    print("   ❌ Marked as NOT RELEVANT")
                    break
                elif response == 's':
                    print("   ⏭️ Skipped")
                    break
                elif response == 'q':
                    print("   ⏹️ Quitting this query")
                    break
                else:
                    print("   Invalid input. Enter y, n, s, or q")
            
            if response == 'q':
                break
        
        # Save after each query
        save_judgments(all_judgments, output_file)
        print(f"\n💾 Progress saved to {output_file}")
    
    return all_judgments


def save_judgments(judgments: List[Dict], filepath: str):
    """Save judgments to CSV"""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'query', 'doc_id', 'doc_url', 'language', 'relevant', 'annotator', 'notes'
        ])
        writer.writeheader()
        writer.writerows(judgments)


def generate_labeling_file(query_pools: Dict[str, List[Dict]], output_file: str):
    """
    Generate a CSV file for offline labeling.
    User can fill in the 'relevant' column manually.
    """
    print(f"\n📝 Generating labeling file: {output_file}")
    
    rows = []
    for query, docs in query_pools.items():
        for doc in docs:
            rows.append({
                'query': query,
                'doc_id': doc['doc_id'],
                'doc_url': doc['url'],
                'title': doc['title'][:100],
                'body_preview': doc['body'][:200].replace('\n', ' '),
                'language': doc['language'],
                'relevant': '',  # TO BE FILLED BY HUMAN
                'annotator': '',
                'notes': f"methods: {','.join(doc['_methods'])}"
            })
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'query', 'doc_id', 'doc_url', 'title', 'body_preview', 
            'language', 'relevant', 'annotator', 'notes'
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Generated {output_file} with {len(rows)} rows to label")
    print("\nInstructions:")
    print("  1. Open the CSV file in Excel/Google Sheets")
    print("  2. For each row, fill 'relevant' column with 'yes' or 'no'")
    print("  3. Add your name in 'annotator' column")
    print("  4. Save and run the evaluation script")


def main():
    # Configuration
    DATASET_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dataset', 'articles_all.jsonl'
    )
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Test queries - these should be meaningful queries for your domain
    TEST_QUERIES = [
        "নির্বাচন",           # election
        "বিএনপি",             # BNP (political party)
        "ঢাকা",               # Dhaka
        "ক্রিকেট",            # cricket
        "বাংলাদেশ সরকার",      # Bangladesh government
    ]
    
    print("=" * 70)
    print("🔧 PROPER RELEVANCE JUDGMENT CREATION")
    print("=" * 70)
    
    # Load documents
    print(f"\n📂 Loading documents from {DATASET_PATH}")
    documents = load_documents(DATASET_PATH, limit=100)
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create document pools
    query_pools = create_document_pool(documents, TEST_QUERIES, pool_depth=15)
    
    # Ask user which mode
    print("\n" + "=" * 70)
    print("Choose labeling mode:")
    print("  1. Interactive (label in terminal now)")
    print("  2. Generate CSV file (label offline in spreadsheet)")
    print("=" * 70)
    
    mode = input("\nEnter 1 or 2: ").strip()
    
    if mode == '1':
        output_file = os.path.join(OUTPUT_DIR, 'human_relevance_judgments.csv')
        interactive_labeling(query_pools, output_file)
    else:
        output_file = os.path.join(OUTPUT_DIR, 'relevance_labeling_template.csv')
        generate_labeling_file(query_pools, output_file)
    
    print("\n" + "=" * 70)
    print("🎉 DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
