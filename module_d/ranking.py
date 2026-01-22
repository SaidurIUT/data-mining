"""
Module D - Ranking & Scoring System
====================================
Implements ranking functions with normalized scores (0-1), 
top-K document retrieval, execution time tracking, and low-confidence warnings.
"""

import time
import json
import sys
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RetrievalMethod(Enum):
    """Available retrieval methods"""
    BM25 = "bm25"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class TimingBreakdown:
    """Timing breakdown for query execution"""
    total_ms: float = 0.0
    preprocessing_ms: float = 0.0
    embedding_ms: float = 0.0
    ranking_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_ms": round(self.total_ms, 2),
            "preprocessing_ms": round(self.preprocessing_ms, 2),
            "embedding_ms": round(self.embedding_ms, 2),
            "ranking_ms": round(self.ranking_ms, 2)
        }
    
    def __str__(self) -> str:
        return (f"Total: {self.total_ms:.2f}ms | "
                f"Preprocess: {self.preprocessing_ms:.2f}ms | "
                f"Embedding: {self.embedding_ms:.2f}ms | "
                f"Ranking: {self.ranking_ms:.2f}ms")


@dataclass
class RankedResult:
    """A single ranked result with normalized score"""
    rank: int
    doc_id: int
    score: float  # Normalized to [0, 1]
    document: Dict
    raw_scores: Dict = field(default_factory=dict)  # Original scores from each method
    
    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "title": self.document.get('title', 'N/A'),
            "url": self.document.get('url', 'N/A'),
            "language": self.document.get('language', 'N/A'),
            "raw_scores": {k: round(v, 4) for k, v in self.raw_scores.items()}
        }


@dataclass
class QueryResult:
    """Complete result for a query including timing and warnings"""
    query: str
    method: RetrievalMethod
    results: List[RankedResult]
    timing: TimingBreakdown
    warnings: List[str] = field(default_factory=list)
    top_score: float = 0.0
    
    def has_low_confidence(self, threshold: float = 0.20) -> bool:
        return self.top_score < threshold
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "method": self.method.value,
            "top_score": round(self.top_score, 4),
            "num_results": len(self.results),
            "timing": self.timing.to_dict(),
            "warnings": self.warnings,
            "results": [r.to_dict() for r in self.results]
        }


class ScoreNormalizer:
    """Normalizes scores from different retrieval methods to [0, 1] range"""
    
    @staticmethod
    def min_max_normalize(scores: List[float], min_val: float = None, max_val: float = None) -> List[float]:
        """Min-max normalization to [0, 1]"""
        if not scores:
            return []
        
        if min_val is None:
            min_val = min(scores)
        if max_val is None:
            max_val = max(scores)
        
        if max_val == min_val:
            return [1.0 if s == max_val else 0.0 for s in scores]
        
        return [(s - min_val) / (max_val - min_val) for s in scores]
    
    @staticmethod
    def normalize_bm25(scores: List[float]) -> List[float]:
        """
        Normalize BM25 scores to [0, 1]
        BM25 scores are typically unbounded, so we use min-max within result set
        """
        if not scores:
            return []
        
        # BM25 scores can be negative in some implementations
        # We shift to make minimum 0, then normalize
        min_score = min(scores)
        if min_score < 0:
            scores = [s - min_score for s in scores]
        
        return ScoreNormalizer.min_max_normalize(scores)
    
    @staticmethod
    def normalize_fuzzy(scores: List[float]) -> List[float]:
        """
        Normalize fuzzy scores to [0, 1]
        Fuzzy scores are typically 0-100 (percentage)
        """
        return [s / 100.0 for s in scores]
    
    @staticmethod
    def normalize_semantic(scores: List[float]) -> List[float]:
        """
        Normalize semantic similarity scores to [0, 1]
        Cosine similarity is already in [-1, 1], we map to [0, 1]
        """
        return [(s + 1) / 2 if s < 0 else s for s in scores]


class RankingSystem:
    """
    Main ranking system that wraps retrieval methods and provides:
    - Normalized scores (0-1)
    - Top-K document ranking
    - Query execution timing
    - Low-confidence warnings
    """
    
    def __init__(self, documents: List[Dict], confidence_threshold: float = 0.20):
        """
        Initialize the ranking system
        
        Args:
            documents: List of documents (must have 'title' and 'body' fields)
            confidence_threshold: Threshold below which low-confidence warning is shown
        """
        self.documents = documents
        self.confidence_threshold = confidence_threshold
        self.normalizer = ScoreNormalizer()
        
        # Lazy-loaded retrievers
        self._bm25 = None
        self._fuzzy = None
        self._semantic = None
        self._hybrid = None
        
        print(f"✅ RankingSystem initialized with {len(documents)} documents")
        print(f"   Confidence threshold: {confidence_threshold}")
    
    def _get_bm25(self):
        """Lazy load BM25 retriever"""
        if self._bm25 is None:
            from module_c.bm25 import BM25Search
            self._bm25 = BM25Search(self.documents)
        return self._bm25
    
    def _get_fuzzy(self):
        """Lazy load Fuzzy retriever"""
        if self._fuzzy is None:
            from module_c.fuzzy import FuzzySearch
            self._fuzzy = FuzzySearch(self.documents, threshold=50)
        return self._fuzzy
    
    def _get_semantic(self):
        """Lazy load Semantic retriever"""
        if self._semantic is None:
            from module_c.semantic import SemanticSearch
            self._semantic = SemanticSearch(self.documents)
        return self._semantic
    
    def _get_hybrid(self):
        """Lazy load Hybrid retriever"""
        if self._hybrid is None:
            from module_c.hybrid import HybridSearch
            self._hybrid = HybridSearch(self.documents)
        return self._hybrid
    
    def search(
        self,
        query: str,
        method: RetrievalMethod = RetrievalMethod.HYBRID,
        k: int = 10,
        verbose: bool = True
    ) -> QueryResult:
        """
        Execute a search query and return ranked, normalized results
        
        Args:
            query: Search query string
            method: Retrieval method to use
            k: Number of top results to return
            verbose: Whether to print results
        
        Returns:
            QueryResult with ranked results, timing, and warnings
        """
        timing = TimingBreakdown()
        start_total = time.time()
        
        # Preprocessing time
        start_preprocess = time.time()
        query = query.strip()
        timing.preprocessing_ms = (time.time() - start_preprocess) * 1000
        
        # Execute search based on method
        start_embedding = time.time()
        raw_results = self._execute_search(query, method, k * 2)  # Get more for better ranking
        timing.embedding_ms = (time.time() - start_embedding) * 1000
        
        # Rank and normalize
        start_ranking = time.time()
        ranked_results = self._rank_and_normalize(raw_results, method, k)
        timing.ranking_ms = (time.time() - start_ranking) * 1000
        
        timing.total_ms = (time.time() - start_total) * 1000
        
        # Build query result
        top_score = ranked_results[0].score if ranked_results else 0.0
        warnings = []
        
        if top_score < self.confidence_threshold:
            warnings.append(
                f"⚠️ Warning: Retrieved results may not be relevant. "
                f"Matching confidence is low (score: {top_score:.2f}). "
                f"Consider rephrasing your query or checking translation quality."
            )
        
        result = QueryResult(
            query=query,
            method=method,
            results=ranked_results,
            timing=timing,
            warnings=warnings,
            top_score=top_score
        )
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _execute_search(
        self,
        query: str,
        method: RetrievalMethod,
        k: int
    ) -> List[Tuple[int, float, Dict]]:
        """Execute search using specified method"""
        
        if method == RetrievalMethod.BM25:
            retriever = self._get_bm25()
            return retriever.search(query, k=k)
        
        elif method == RetrievalMethod.FUZZY:
            retriever = self._get_fuzzy()
            return retriever.search(query, k=k)
        
        elif method == RetrievalMethod.SEMANTIC:
            retriever = self._get_semantic()
            return retriever.search(query, k=k)
        
        elif method == RetrievalMethod.HYBRID:
            retriever = self._get_hybrid()
            return retriever.search(query, k=k, verbose=False)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _rank_and_normalize(
        self,
        raw_results: List,
        method: RetrievalMethod,
        k: int
    ) -> List[RankedResult]:
        """Normalize scores and create ranked results"""
        
        if not raw_results:
            return []
        
        # Handle different result formats
        # BM25, Fuzzy, Semantic: (doc_id, score, doc)
        # Hybrid: (doc_id, score, doc, breakdown)
        is_hybrid = method == RetrievalMethod.HYBRID
        
        # Extract scores
        scores = [r[1] for r in raw_results]
        
        # Normalize based on method
        if method == RetrievalMethod.BM25:
            normalized_scores = self.normalizer.normalize_bm25(scores)
        elif method == RetrievalMethod.FUZZY:
            normalized_scores = self.normalizer.normalize_fuzzy(scores)
        elif method == RetrievalMethod.SEMANTIC:
            normalized_scores = self.normalizer.normalize_semantic(scores)
        elif method == RetrievalMethod.HYBRID:
            # Hybrid already outputs normalized scores
            normalized_scores = scores
        else:
            normalized_scores = self.normalizer.min_max_normalize(scores)
        
        # Create ranked results
        ranked = []
        for i, (result, norm_score) in enumerate(zip(raw_results, normalized_scores)):
            if i >= k:
                break
            
            # Unpack based on format
            if is_hybrid and len(result) == 4:
                doc_id, raw_score, doc, breakdown = result
                raw_scores = breakdown
            else:
                doc_id, raw_score, doc = result[:3]
                raw_scores = {method.value: raw_score}
            
            ranked.append(RankedResult(
                rank=i + 1,
                doc_id=doc_id,
                score=norm_score,
                document=doc,
                raw_scores=raw_scores
            ))
        
        # Sort by normalized score (should already be sorted, but ensure)
        ranked.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks after sorting
        for i, r in enumerate(ranked):
            r.rank = i + 1
        
        return ranked
    
    def _print_results(self, result: QueryResult):
        """Pretty print search results"""
        print("\n" + "=" * 70)
        print(f"🔍 Query: '{result.query}'")
        print(f"📊 Method: {result.method.value.upper()}")
        print("=" * 70)
        
        # Timing breakdown
        print(f"\n⏱️  Execution Time: {result.timing}")
        
        # Warnings
        for warning in result.warnings:
            print(f"\n{warning}")
        
        # Results
        if not result.results:
            print("\n❌ No results found")
            return
        
        print(f"\n📋 Top {len(result.results)} Results (scores normalized to 0-1):")
        print("-" * 70)
        
        for r in result.results:
            title = r.document.get('title', 'N/A')[:55]
            lang = r.document.get('language', '?')[:2]
            confidence = "🟢" if r.score >= 0.5 else "🟡" if r.score >= 0.2 else "🔴"
            
            print(f"{r.rank:2}. {confidence} Score: {r.score:.4f} | [{lang}] {title}...")
        
        print("-" * 70)
        print(f"✅ Top confidence score: {result.top_score:.4f}")
    
    def search_all_methods(
        self,
        query: str,
        k: int = 10,
        verbose: bool = True
    ) -> Dict[RetrievalMethod, QueryResult]:
        """
        Execute search with all methods and compare results
        
        Args:
            query: Search query
            k: Number of results per method
            verbose: Print comparison
        
        Returns:
            Dictionary mapping method to QueryResult
        """
        results = {}
        
        for method in RetrievalMethod:
            try:
                results[method] = self.search(query, method, k, verbose=False)
            except Exception as e:
                print(f"❌ {method.value} failed: {e}")
        
        if verbose:
            self._print_comparison(query, results)
        
        return results
    
    def _print_comparison(self, query: str, results: Dict[RetrievalMethod, QueryResult]):
        """Print comparison of results from all methods"""
        print("\n" + "=" * 80)
        print(f"🔍 MULTI-METHOD COMPARISON: '{query}'")
        print("=" * 80)
        
        # Timing comparison
        print("\n⏱️  Timing Comparison:")
        print("-" * 40)
        for method, result in results.items():
            print(f"   {method.value:10} : {result.timing.total_ms:8.2f} ms")
        
        # Top result comparison
        print("\n🏆 Top Result Comparison:")
        print("-" * 80)
        
        for method, result in results.items():
            if result.results:
                top = result.results[0]
                title = top.document.get('title', 'N/A')[:40]
                print(f"   {method.value:10} : Score={top.score:.4f} | {title}...")
            else:
                print(f"   {method.value:10} : No results")
        
        # Warnings
        for method, result in results.items():
            for warning in result.warnings:
                print(f"\n[{method.value}] {warning}")


def load_documents(filepath: str, limit: int = None) -> List[Dict]:
    """Load documents from JSONL file"""
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            documents.append(json.loads(line))
    return documents


# Demo usage
if __name__ == "__main__":
    # Load dataset
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dataset', 'articles_all.jsonl'
    )
    
    print("📂 Loading documents...")
    documents = load_documents(dataset_path, limit=100)
    print(f"✅ Loaded {len(documents)} documents\n")
    
    # Initialize ranking system
    ranker = RankingSystem(documents, confidence_threshold=0.20)
    
    # Test queries
    test_queries = [
        "নির্বাচন",  # Bangla: election
        "Bangladesh election",  # English
        "xyz123abc",  # Nonsense - should trigger low confidence
    ]
    
    for query in test_queries:
        result = ranker.search(query, method=RetrievalMethod.HYBRID, k=5)
        print("\n")
    
    # Compare all methods
    print("\n" + "=" * 80)
    print("COMPARING ALL METHODS")
    print("=" * 80)
    ranker.search_all_methods("নির্বাচন ভোট", k=5)
