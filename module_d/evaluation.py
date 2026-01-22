"""
Module D - Evaluation Metrics
==============================
Implements standard IR evaluation metrics:
- Precision@K
- Recall@K
- nDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)

Also includes comparison with external search engines.
"""

import json
import csv
import os
import math
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RelevanceJudgment:
    """A single relevance judgment for a query-document pair"""
    query: str
    doc_url: str
    doc_id: int
    language: str
    relevant: bool
    annotator: str = "default"
    notes: str = ""


@dataclass
class EvaluationResult:
    """Evaluation results for a single query"""
    query: str
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    num_relevant: int = 0
    num_retrieved: int = 0
    relevant_retrieved: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "precision@10": round(self.precision_at_k.get(10, 0), 4),
            "recall@50": round(self.recall_at_k.get(50, 0), 4),
            "ndcg@10": round(self.ndcg_at_k.get(10, 0), 4),
            "mrr": round(self.mrr, 4),
            "num_relevant": self.num_relevant,
            "num_retrieved": self.num_retrieved,
            "relevant_retrieved": self.relevant_retrieved
        }


@dataclass
class SystemEvaluation:
    """Overall system evaluation across multiple queries"""
    query_results: List[EvaluationResult] = field(default_factory=list)
    
    # Aggregate metrics
    mean_precision_at_10: float = 0.0
    mean_recall_at_50: float = 0.0
    mean_ndcg_at_10: float = 0.0
    mean_mrr: float = 0.0
    
    # Targets
    precision_target: float = 0.6
    recall_target: float = 0.5
    ndcg_target: float = 0.5
    mrr_target: float = 0.4
    
    def compute_aggregates(self):
        """Compute mean metrics across all queries"""
        if not self.query_results:
            return
        
        n = len(self.query_results)
        self.mean_precision_at_10 = sum(r.precision_at_k.get(10, 0) for r in self.query_results) / n
        self.mean_recall_at_50 = sum(r.recall_at_k.get(50, 0) for r in self.query_results) / n
        self.mean_ndcg_at_10 = sum(r.ndcg_at_k.get(10, 0) for r in self.query_results) / n
        self.mean_mrr = sum(r.mrr for r in self.query_results) / n
    
    def meets_targets(self) -> Dict[str, bool]:
        """Check if system meets all target metrics"""
        return {
            "precision@10": self.mean_precision_at_10 >= self.precision_target,
            "recall@50": self.mean_recall_at_50 >= self.recall_target,
            "ndcg@10": self.mean_ndcg_at_10 >= self.ndcg_target,
            "mrr": self.mean_mrr >= self.mrr_target
        }
    
    def to_dict(self) -> Dict:
        targets = self.meets_targets()
        return {
            "num_queries": len(self.query_results),
            "metrics": {
                "precision@10": {
                    "value": round(self.mean_precision_at_10, 4),
                    "target": self.precision_target,
                    "meets_target": targets["precision@10"]
                },
                "recall@50": {
                    "value": round(self.mean_recall_at_50, 4),
                    "target": self.recall_target,
                    "meets_target": targets["recall@50"]
                },
                "ndcg@10": {
                    "value": round(self.mean_ndcg_at_10, 4),
                    "target": self.ndcg_target,
                    "meets_target": targets["ndcg@10"]
                },
                "mrr": {
                    "value": round(self.mean_mrr, 4),
                    "target": self.mrr_target,
                    "meets_target": targets["mrr"]
                }
            },
            "query_results": [r.to_dict() for r in self.query_results]
        }


class RelevanceLabeler:
    """
    Tool for creating and managing relevance judgments.
    Saves/loads from CSV with columns: query, doc_url, doc_id, language, relevant, annotator, notes
    """
    
    def __init__(self, filepath: str = "relevance_judgments.csv"):
        self.filepath = filepath
        self.judgments: List[RelevanceJudgment] = []
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing judgments from CSV"""
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.judgments.append(RelevanceJudgment(
                        query=row['query'],
                        doc_url=row['doc_url'],
                        doc_id=int(row['doc_id']),
                        language=row['language'],
                        relevant=row['relevant'].lower() == 'yes',
                        annotator=row.get('annotator', 'default'),
                        notes=row.get('notes', '')
                    ))
            print(f"📂 Loaded {len(self.judgments)} existing judgments from {self.filepath}")
    
    def add_judgment(
        self,
        query: str,
        doc_url: str,
        doc_id: int,
        language: str,
        relevant: bool,
        annotator: str = "default",
        notes: str = ""
    ):
        """Add a relevance judgment"""
        self.judgments.append(RelevanceJudgment(
            query=query,
            doc_url=doc_url,
            doc_id=doc_id,
            language=language,
            relevant=relevant,
            annotator=annotator,
            notes=notes
        ))
    
    def save(self):
        """Save judgments to CSV"""
        with open(self.filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'query', 'doc_url', 'doc_id', 'language', 'relevant', 'annotator', 'notes'
            ])
            writer.writeheader()
            for j in self.judgments:
                writer.writerow({
                    'query': j.query,
                    'doc_url': j.doc_url,
                    'doc_id': j.doc_id,
                    'language': j.language,
                    'relevant': 'yes' if j.relevant else 'no',
                    'annotator': j.annotator,
                    'notes': j.notes
                })
        print(f"💾 Saved {len(self.judgments)} judgments to {self.filepath}")
    
    def get_relevant_docs(self, query: str) -> Set[int]:
        """Get set of relevant document IDs for a query"""
        return {j.doc_id for j in self.judgments if j.query == query and j.relevant}
    
    def get_all_queries(self) -> List[str]:
        """Get list of all unique queries with judgments"""
        return list(set(j.query for j in self.judgments))


class Evaluator:
    """
    Evaluation system for measuring IR performance.
    Implements Precision@K, Recall@K, nDCG@K, and MRR.
    """
    
    def __init__(self, relevance_labeler: RelevanceLabeler = None):
        """
        Initialize evaluator
        
        Args:
            relevance_labeler: RelevanceLabeler with relevance judgments
        """
        self.labeler = relevance_labeler or RelevanceLabeler()
    
    def precision_at_k(self, retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """
        Calculate Precision@K
        
        Precision@K = (# relevant docs in top K) / K
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranked order
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Precision@K score (0 to 1)
        """
        if k <= 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_in_top_k = len([doc_id for doc_id in top_k if doc_id in relevant_ids])
        
        return relevant_in_top_k / k
    
    def recall_at_k(self, retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """
        Calculate Recall@K
        
        Recall@K = (# relevant docs retrieved in top K) / (total # relevant docs)
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranked order
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Recall@K score (0 to 1)
        """
        if not relevant_ids:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_in_top_k = len([doc_id for doc_id in top_k if doc_id in relevant_ids])
        
        return relevant_in_top_k / len(relevant_ids)
    
    def dcg_at_k(self, retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at K
        
        DCG@K = Σ (rel_i / log2(i+1)) for i = 1 to K
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranked order
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1 if doc_id in relevant_ids else 0
            # i+2 because log2(1) = 0, we use log2(rank+1) where rank starts at 1
            dcg += rel / math.log2(i + 2)
        
        return dcg
    
    def ndcg_at_k(self, retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K
        
        nDCG@K = DCG@K / IDCG@K
        
        Where IDCG is the ideal DCG (if all relevant docs were at top)
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranked order
            relevant_ids: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            nDCG@K score (0 to 1)
        """
        dcg = self.dcg_at_k(retrieved_ids, relevant_ids, k)
        
        # Calculate ideal DCG (all relevant docs at top)
        num_relevant = min(len(relevant_ids), k)
        idcg = sum(1 / math.log2(i + 2) for i in range(num_relevant))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def reciprocal_rank(self, retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
        """
        Calculate Reciprocal Rank
        
        RR = 1 / (rank of first relevant document)
        
        Args:
            retrieved_ids: List of retrieved document IDs in ranked order
            relevant_ids: Set of relevant document IDs
        
        Returns:
            Reciprocal Rank score (0 to 1)
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate_query(
        self,
        query: str,
        retrieved_ids: List[int],
        relevant_ids: Set[int] = None,
        k_values: List[int] = [5, 10, 20, 50]
    ) -> EvaluationResult:
        """
        Evaluate a single query
        
        Args:
            query: The query string
            retrieved_ids: List of retrieved document IDs in ranked order
            relevant_ids: Set of relevant document IDs (if None, uses labeler)
            k_values: List of K values to compute metrics for
        
        Returns:
            EvaluationResult with all metrics
        """
        if relevant_ids is None:
            relevant_ids = self.labeler.get_relevant_docs(query)
        
        result = EvaluationResult(query=query)
        result.num_relevant = len(relevant_ids)
        result.num_retrieved = len(retrieved_ids)
        result.relevant_retrieved = len([d for d in retrieved_ids if d in relevant_ids])
        
        # Compute metrics at each K
        for k in k_values:
            result.precision_at_k[k] = self.precision_at_k(retrieved_ids, relevant_ids, k)
            result.recall_at_k[k] = self.recall_at_k(retrieved_ids, relevant_ids, k)
            result.ndcg_at_k[k] = self.ndcg_at_k(retrieved_ids, relevant_ids, k)
        
        # MRR
        result.mrr = self.reciprocal_rank(retrieved_ids, relevant_ids)
        
        return result
    
    def evaluate_system(
        self,
        ranking_system,
        queries: List[str] = None,
        method=None,
        k: int = 50,
        verbose: bool = True
    ) -> SystemEvaluation:
        """
        Evaluate the entire retrieval system across multiple queries
        
        Args:
            ranking_system: RankingSystem instance
            queries: List of queries to evaluate (if None, uses all labeled queries)
            method: Retrieval method to use
            k: Number of results to retrieve per query
            verbose: Print evaluation progress
        
        Returns:
            SystemEvaluation with aggregate metrics
        """
        from module_d.ranking import RetrievalMethod
        
        if method is None:
            method = RetrievalMethod.HYBRID
        
        if queries is None:
            queries = self.labeler.get_all_queries()
        
        if not queries:
            print("⚠️ No queries to evaluate. Add relevance judgments first.")
            return SystemEvaluation()
        
        evaluation = SystemEvaluation()
        
        if verbose:
            print("\n" + "=" * 70)
            print("📊 SYSTEM EVALUATION")
            print("=" * 70)
            print(f"Method: {method.value}")
            print(f"Queries: {len(queries)}")
            print("-" * 70)
        
        for query in queries:
            # Run search
            result = ranking_system.search(query, method=method, k=k, verbose=False)
            
            # Get retrieved doc IDs
            retrieved_ids = [r.doc_id for r in result.results]
            
            # Evaluate
            eval_result = self.evaluate_query(query, retrieved_ids)
            evaluation.query_results.append(eval_result)
            
            if verbose:
                status = "✅" if eval_result.precision_at_k.get(10, 0) >= 0.6 else "⚠️"
                print(f"{status} Query: '{query[:30]}...' | P@10={eval_result.precision_at_k.get(10, 0):.2f} | "
                      f"R@50={eval_result.recall_at_k.get(50, 0):.2f} | nDCG@10={eval_result.ndcg_at_k.get(10, 0):.2f}")
        
        # Compute aggregates
        evaluation.compute_aggregates()
        
        if verbose:
            self._print_evaluation_summary(evaluation)
        
        return evaluation
    
    def _print_evaluation_summary(self, evaluation: SystemEvaluation):
        """Print evaluation summary"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        targets = evaluation.meets_targets()
        
        metrics = [
            ("Precision@10", evaluation.mean_precision_at_10, evaluation.precision_target, targets["precision@10"]),
            ("Recall@50", evaluation.mean_recall_at_50, evaluation.recall_target, targets["recall@50"]),
            ("nDCG@10", evaluation.mean_ndcg_at_10, evaluation.ndcg_target, targets["ndcg@10"]),
            ("MRR", evaluation.mean_mrr, evaluation.mrr_target, targets["mrr"]),
        ]
        
        print(f"\n{'Metric':<15} {'Value':>10} {'Target':>10} {'Status':>10}")
        print("-" * 50)
        
        for name, value, target, meets in metrics:
            status = "✅ PASS" if meets else "❌ FAIL"
            print(f"{name:<15} {value:>10.4f} {target:>10.2f} {status:>10}")
        
        print("-" * 50)
        all_pass = all(targets.values())
        print(f"\n{'🎉 ALL TARGETS MET!' if all_pass else '⚠️ Some targets not met'}")


class SearchEngineComparator:
    """
    Compare retrieval results with external search engines.
    Records comparison data for analysis.
    """
    
    def __init__(self, output_dir: str = "comparisons"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.comparisons = []
    
    def record_comparison(
        self,
        query: str,
        our_results: List[Dict],
        google_results: List[str] = None,
        bing_results: List[str] = None,
        duckduckgo_results: List[str] = None,
        ai_engine_results: List[str] = None,
        notes: str = ""
    ):
        """
        Record a comparison between our system and external search engines
        
        Args:
            query: The search query
            our_results: Our system's top results (list of doc dicts)
            google_results: URLs from Google search
            bing_results: URLs from Bing search
            duckduckgo_results: URLs from DuckDuckGo search
            ai_engine_results: Results from AI-powered engines (ChatGPT, Perplexity, etc.)
            notes: Analysis notes
        """
        comparison = {
            "query": query,
            "our_results": [
                {"title": r.get("title", "N/A")[:100], "url": r.get("url", "N/A")}
                for r in our_results[:5]
            ],
            "google_results": google_results or [],
            "bing_results": bing_results or [],
            "duckduckgo_results": duckduckgo_results or [],
            "ai_engine_results": ai_engine_results or [],
            "notes": notes
        }
        self.comparisons.append(comparison)
    
    def save_comparisons(self, filename: str = "search_engine_comparisons.json"):
        """Save all comparisons to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.comparisons, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(self.comparisons)} comparisons to {filepath}")
    
    def generate_comparison_template(self, queries: List[str], filename: str = "comparison_template.md"):
        """
        Generate a markdown template for manual comparison with search engines
        
        Args:
            queries: List of queries to compare
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Search Engine Comparison Template\n\n")
            f.write("Compare our CLIR system results with popular search engines.\n\n")
            
            for i, query in enumerate(queries, 1):
                f.write(f"## Query {i}: `{query}`\n\n")
                f.write("### Our System Results\n")
                f.write("| Rank | Title | Relevant? |\n")
                f.write("|------|-------|----------|\n")
                for j in range(5):
                    f.write(f"| {j+1} | [Fill in] | Yes/No |\n")
                f.write("\n")
                
                f.write("### Google Results\n")
                f.write("| Rank | URL/Title | Relevant? |\n")
                f.write("|------|-----------|----------|\n")
                for j in range(5):
                    f.write(f"| {j+1} | [Fill in] | Yes/No |\n")
                f.write("\n")
                
                f.write("### DuckDuckGo Results\n")
                f.write("| Rank | URL/Title | Relevant? |\n")
                f.write("|------|-----------|----------|\n")
                for j in range(5):
                    f.write(f"| {j+1} | [Fill in] | Yes/No |\n")
                f.write("\n")
                
                f.write("### Notes\n")
                f.write("_Add analysis notes here_\n\n")
                f.write("---\n\n")
        
        print(f"📝 Generated comparison template: {filepath}")


# Demo usage
if __name__ == "__main__":
    print("=" * 70)
    print("Module D - Evaluation Demo")
    print("=" * 70)
    
    # Create sample relevance judgments
    labeler = RelevanceLabeler("sample_judgments.csv")
    
    # Add some sample judgments (you would do this manually for real evaluation)
    sample_judgments = [
        ("নির্বাচন", "url1", 0, "bn", True),
        ("নির্বাচন", "url2", 1, "bn", True),
        ("নির্বাচন", "url3", 2, "bn", False),
        ("নির্বাচন", "url4", 3, "bn", True),
        ("নির্বাচন", "url5", 4, "bn", False),
        ("বিএনপি", "url10", 10, "bn", True),
        ("বিএনপি", "url11", 11, "bn", True),
        ("বিএনপি", "url12", 12, "bn", False),
    ]
    
    for query, url, doc_id, lang, relevant in sample_judgments:
        labeler.add_judgment(query, url, doc_id, lang, relevant, "demo")
    
    labeler.save()
    
    # Initialize evaluator
    evaluator = Evaluator(labeler)
    
    # Demo: Evaluate a mock retrieval
    print("\n📊 Demo Evaluation:")
    retrieved = [0, 2, 1, 5, 3, 6, 4, 7, 8, 9]  # Mock retrieved doc IDs
    relevant = labeler.get_relevant_docs("নির্বাচন")
    
    result = evaluator.evaluate_query("নির্বাচন", retrieved, relevant)
    print(f"   Precision@10: {result.precision_at_k[10]:.4f}")
    print(f"   Recall@50: {result.recall_at_k[50]:.4f}")
    print(f"   nDCG@10: {result.ndcg_at_k[10]:.4f}")
    print(f"   MRR: {result.mrr:.4f}")
