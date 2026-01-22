"""
Generate REAL evaluation metrics for Module E report.
Runs actual evaluation for each retrieval method.
"""

import json
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_documents(filepath, limit=100):
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            documents.append(json.loads(line))
    return documents


def main():
    from module_d.ranking import RankingSystem, RetrievalMethod
    from module_d.evaluation import Evaluator, RelevanceLabeler
    
    # Load documents
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dataset', 'articles_all.jsonl'
    )
    documents = load_documents(dataset_path, limit=100)
    print(f"Loaded {len(documents)} documents")
    
    # Load existing relevance judgments
    judgment_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'module_d', 'proper_relevance_judgments.csv'
    )
    
    labeler = RelevanceLabeler(judgment_file)
    queries = labeler.get_all_queries()
    print(f"Queries to evaluate: {queries}")
    
    # Initialize ranking system
    ranker = RankingSystem(documents, confidence_threshold=0.20)
    evaluator = Evaluator(labeler)
    
    # Store all results
    all_results = []
    method_aggregates = {}
    
    # Evaluate each method
    for method in [RetrievalMethod.BM25, RetrievalMethod.SEMANTIC, RetrievalMethod.HYBRID]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method.value.upper()}")
        print(f"{'='*60}")
        
        method_results = {
            'p10': [], 'r50': [], 'ndcg10': [], 'mrr': [], 'time': []
        }
        
        for query in queries:
            # Run search with timing
            import time
            start = time.time()
            result = ranker.search(query, method=method, k=50, verbose=False)
            elapsed_ms = (time.time() - start) * 1000
            
            # Get retrieved doc IDs
            retrieved_ids = [r.doc_id for r in result.results]
            
            # Evaluate
            eval_result = evaluator.evaluate_query(query, retrieved_ids)
            
            # Store
            row = {
                'query': query,
                'method': method.value,
                'precision_at_10': round(eval_result.precision_at_k.get(10, 0), 4),
                'recall_at_50': round(eval_result.recall_at_k.get(50, 0), 4),
                'ndcg_at_10': round(eval_result.ndcg_at_k.get(10, 0), 4),
                'mrr': round(eval_result.mrr, 4),
                'execution_time_ms': round(elapsed_ms, 2),
                'num_results': len(result.results),
                'top_score': round(result.top_score, 4)
            }
            all_results.append(row)
            
            method_results['p10'].append(eval_result.precision_at_k.get(10, 0))
            method_results['r50'].append(eval_result.recall_at_k.get(50, 0))
            method_results['ndcg10'].append(eval_result.ndcg_at_k.get(10, 0))
            method_results['mrr'].append(eval_result.mrr)
            method_results['time'].append(elapsed_ms)
            
            print(f"  {query}: P@10={row['precision_at_10']}, R@50={row['recall_at_50']}, "
                  f"nDCG@10={row['ndcg_at_10']}, MRR={row['mrr']}, Time={row['execution_time_ms']:.0f}ms")
        
        # Calculate aggregates
        n = len(queries)
        method_aggregates[method.value] = {
            'precision_at_10': round(sum(method_results['p10']) / n, 4),
            'recall_at_50': round(sum(method_results['r50']) / n, 4),
            'ndcg_at_10': round(sum(method_results['ndcg10']) / n, 4),
            'mrr': round(sum(method_results['mrr']) / n, 4),
            'avg_time_ms': round(sum(method_results['time']) / n, 2)
        }
        
        print(f"\n  AVERAGES: P@10={method_aggregates[method.value]['precision_at_10']}, "
              f"R@50={method_aggregates[method.value]['recall_at_50']}, "
              f"nDCG@10={method_aggregates[method.value]['ndcg_at_10']}")
    
    # Save detailed results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Per-query results
    with open(os.path.join(output_dir, 'evaluation_results_REAL.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'query', 'method', 'precision_at_10', 'recall_at_50', 'ndcg_at_10', 
            'mrr', 'execution_time_ms', 'num_results', 'top_score'
        ])
        writer.writeheader()
        writer.writerows(all_results)
    
    # Model comparison (aggregates)
    with open(os.path.join(output_dir, 'model_comparison_REAL.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'bm25', 'semantic', 'hybrid', 'target'])
        
        writer.writerow(['precision_at_10', 
                        method_aggregates['bm25']['precision_at_10'],
                        method_aggregates['semantic']['precision_at_10'],
                        method_aggregates['hybrid']['precision_at_10'],
                        0.60])
        writer.writerow(['recall_at_50',
                        method_aggregates['bm25']['recall_at_50'],
                        method_aggregates['semantic']['recall_at_50'],
                        method_aggregates['hybrid']['recall_at_50'],
                        0.50])
        writer.writerow(['ndcg_at_10',
                        method_aggregates['bm25']['ndcg_at_10'],
                        method_aggregates['semantic']['ndcg_at_10'],
                        method_aggregates['hybrid']['ndcg_at_10'],
                        0.50])
        writer.writerow(['mrr',
                        method_aggregates['bm25']['mrr'],
                        method_aggregates['semantic']['mrr'],
                        method_aggregates['hybrid']['mrr'],
                        0.40])
        writer.writerow(['avg_time_ms',
                        method_aggregates['bm25']['avg_time_ms'],
                        method_aggregates['semantic']['avg_time_ms'],
                        method_aggregates['hybrid']['avg_time_ms'],
                        'N/A'])
    
    print(f"\n{'='*60}")
    print("REAL DATA SAVED:")
    print(f"  - evaluation_results_REAL.csv")
    print(f"  - model_comparison_REAL.csv")
    print(f"{'='*60}")
    
    # Print summary for report
    print("\n\n📊 COPY THIS TO YOUR REPORT:")
    print("="*60)
    print("\n### Model Comparison (REAL DATA)\n")
    print("| Metric | BM25 | Semantic | Hybrid | Target |")
    print("|--------|------|----------|--------|--------|")
    print(f"| Precision@10 | {method_aggregates['bm25']['precision_at_10']} | "
          f"{method_aggregates['semantic']['precision_at_10']} | "
          f"{method_aggregates['hybrid']['precision_at_10']} | ≥0.60 |")
    print(f"| Recall@50 | {method_aggregates['bm25']['recall_at_50']} | "
          f"{method_aggregates['semantic']['recall_at_50']} | "
          f"{method_aggregates['hybrid']['recall_at_50']} | ≥0.50 |")
    print(f"| nDCG@10 | {method_aggregates['bm25']['ndcg_at_10']} | "
          f"{method_aggregates['semantic']['ndcg_at_10']} | "
          f"{method_aggregates['hybrid']['ndcg_at_10']} | ≥0.50 |")
    print(f"| MRR | {method_aggregates['bm25']['mrr']} | "
          f"{method_aggregates['semantic']['mrr']} | "
          f"{method_aggregates['hybrid']['mrr']} | ≥0.40 |")
    print(f"| Avg Time (ms) | {method_aggregates['bm25']['avg_time_ms']:.0f} | "
          f"{method_aggregates['semantic']['avg_time_ms']:.0f} | "
          f"{method_aggregates['hybrid']['avg_time_ms']:.0f} | - |")


if __name__ == "__main__":
    main()
