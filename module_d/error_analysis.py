"""
Module D - Error Analysis
==========================
Detailed error analysis for retrieval failures with specific examples:
1. Translation Failures
2. Named Entity Mismatch
3. Semantic vs. Lexical Wins
4. Cross-Script Ambiguity
5. Code-Switching

Each analysis includes query text, retrieved documents, and detailed analysis.
"""

import json
import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ErrorCategory(Enum):
    """Categories of retrieval errors/successes"""
    TRANSLATION_FAILURE = "translation_failure"
    NER_MISMATCH = "ner_mismatch"
    SEMANTIC_WIN = "semantic_vs_lexical_win"
    LEXICAL_WIN = "lexical_vs_semantic_win"
    CROSS_SCRIPT_AMBIGUITY = "cross_script_ambiguity"
    CODE_SWITCHING = "code_switching"


@dataclass
class CaseStudy:
    """A single case study for error analysis"""
    category: ErrorCategory
    query: str
    query_language: str
    expected_behavior: str
    actual_behavior: str
    retrieved_docs: List[Dict] = field(default_factory=list)
    analysis: str = ""
    success: bool = False
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "query": self.query,
            "query_language": self.query_language,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
            "retrieved_docs": [
                {
                    "rank": i + 1,
                    "title": doc.get("title", "N/A")[:100],
                    "url": doc.get("url", "N/A"),
                    "score": doc.get("score", 0)
                }
                for i, doc in enumerate(self.retrieved_docs[:5])
            ],
            "analysis": self.analysis,
            "success": self.success,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report for this case study"""
        status = "✅ SUCCESS" if self.success else "❌ FAILURE"
        
        md = f"""
### Case Study: {self.category.value.replace('_', ' ').title()}

**Status:** {status}

**Query:** `{self.query}` ({self.query_language})

**Expected Behavior:**
{self.expected_behavior}

**Actual Behavior:**
{self.actual_behavior}

**Retrieved Documents:**
| Rank | Title | Score |
|------|-------|-------|
"""
        for i, doc in enumerate(self.retrieved_docs[:5]):
            title = doc.get("title", "N/A")[:50]
            score = doc.get("score", 0)
            md += f"| {i+1} | {title}... | {score:.4f} |\n"
        
        md += f"""
**Analysis:**
{self.analysis}

**Recommendations:**
"""
        for rec in self.recommendations:
            md += f"- {rec}\n"
        
        md += "\n---\n"
        return md


class ErrorAnalyzer:
    """
    Comprehensive error analysis for CLIR retrieval system.
    Analyzes specific failure/success cases and generates detailed reports.
    """
    
    def __init__(self, ranking_system=None, output_dir: str = "error_analysis"):
        """
        Initialize error analyzer
        
        Args:
            ranking_system: RankingSystem instance from module_d
            output_dir: Directory to save analysis reports
        """
        self.ranking_system = ranking_system
        self.output_dir = output_dir
        self.case_studies: List[CaseStudy] = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def set_ranking_system(self, ranking_system):
        """Set the ranking system to analyze"""
        self.ranking_system = ranking_system
    
    def analyze_translation_failure(
        self,
        query: str,
        mistranslation: str,
        correct_translation: str,
        verbose: bool = True
    ) -> CaseStudy:
        """
        Analyze translation failure case
        
        Example: Query "চেয়ার" (chair) mistranslated to "Chairman"
        
        Args:
            query: Original query (e.g., "চেয়ার")
            mistranslation: What it was mistranslated to (e.g., "Chairman")
            correct_translation: What it should be (e.g., "chair")
            verbose: Print analysis
        """
        from module_d.ranking import RetrievalMethod
        
        if verbose:
            print("\n" + "=" * 70)
            print("🔍 TRANSLATION FAILURE ANALYSIS")
            print("=" * 70)
        
        # Search with original query
        result_original = self.ranking_system.search(
            query, method=RetrievalMethod.HYBRID, k=5, verbose=False
        )
        
        # Search with mistranslation
        result_mistrans = self.ranking_system.search(
            mistranslation, method=RetrievalMethod.HYBRID, k=5, verbose=False
        )
        
        # Search with correct translation
        result_correct = self.ranking_system.search(
            correct_translation, method=RetrievalMethod.HYBRID, k=5, verbose=False
        )
        
        # Analyze
        original_titles = [r.document.get('title', '')[:50] for r in result_original.results]
        mistrans_titles = [r.document.get('title', '')[:50] for r in result_mistrans.results]
        correct_titles = [r.document.get('title', '')[:50] for r in result_correct.results]
        
        # Check overlap
        overlap_mistrans = len(set(original_titles) & set(mistrans_titles))
        overlap_correct = len(set(original_titles) & set(correct_titles))
        
        success = overlap_correct > overlap_mistrans
        
        case = CaseStudy(
            category=ErrorCategory.TRANSLATION_FAILURE,
            query=query,
            query_language="bn" if any('\u0980' <= c <= '\u09FF' for c in query) else "en",
            expected_behavior=f"Query '{query}' should be translated to '{correct_translation}' and find relevant documents",
            actual_behavior=f"If mistranslated to '{mistranslation}', retrieves different/wrong documents",
            retrieved_docs=[r.document | {"score": r.score} for r in result_original.results],
            analysis=f"""
Translation Analysis:
- Original query '{query}' retrieved {len(result_original.results)} results
- Mistranslation '{mistranslation}' retrieved {len(result_mistrans.results)} results
- Correct translation '{correct_translation}' retrieved {len(result_correct.results)} results
- Overlap (original vs mistranslation): {overlap_mistrans}/5 documents
- Overlap (original vs correct): {overlap_correct}/5 documents

The semantic model {'successfully handles' if success else 'struggles with'} this translation ambiguity.
""",
            success=success,
            recommendations=[
                "Use multiple translation candidates",
                "Implement translation confidence scoring",
                "Consider context-aware translation",
                "Add domain-specific translation dictionaries"
            ]
        )
        
        self.case_studies.append(case)
        
        if verbose:
            print(case.to_markdown())
        
        return case
    
    def analyze_ner_mismatch(
        self,
        entity_bn: str,
        entity_en: str,
        verbose: bool = True
    ) -> CaseStudy:
        """
        Analyze Named Entity Recognition mismatch
        
        Example: Query mentions "ঢাকা" (Dhaka) but documents use "Dhaka" in English
        
        Args:
            entity_bn: Entity in Bangla (e.g., "ঢাকা")
            entity_en: Entity in English (e.g., "Dhaka")
            verbose: Print analysis
        """
        from module_d.ranking import RetrievalMethod
        
        if verbose:
            print("\n" + "=" * 70)
            print("🔍 NAMED ENTITY MISMATCH ANALYSIS")
            print("=" * 70)
        
        # BM25 search (lexical)
        result_bm25_bn = self.ranking_system.search(
            entity_bn, method=RetrievalMethod.BM25, k=5, verbose=False
        )
        result_bm25_en = self.ranking_system.search(
            entity_en, method=RetrievalMethod.BM25, k=5, verbose=False
        )
        
        # Semantic search
        result_semantic_bn = self.ranking_system.search(
            entity_bn, method=RetrievalMethod.SEMANTIC, k=5, verbose=False
        )
        result_semantic_en = self.ranking_system.search(
            entity_en, method=RetrievalMethod.SEMANTIC, k=5, verbose=False
        )
        
        # Check if semantic can bridge the gap
        bm25_overlap = len(
            set(r.doc_id for r in result_bm25_bn.results) &
            set(r.doc_id for r in result_bm25_en.results)
        )
        semantic_overlap = len(
            set(r.doc_id for r in result_semantic_bn.results) &
            set(r.doc_id for r in result_semantic_en.results)
        )
        
        success = semantic_overlap > bm25_overlap
        
        case = CaseStudy(
            category=ErrorCategory.NER_MISMATCH,
            query=f"{entity_bn} / {entity_en}",
            query_language="bn/en",
            expected_behavior=f"Both '{entity_bn}' (Bangla) and '{entity_en}' (English) should retrieve same entity-related documents",
            actual_behavior=f"BM25 finds {bm25_overlap}/5 common docs, Semantic finds {semantic_overlap}/5 common docs",
            retrieved_docs=[r.document | {"score": r.score} for r in result_semantic_bn.results],
            analysis=f"""
NER Mismatch Analysis:
- Bangla entity: '{entity_bn}'
- English entity: '{entity_en}'

BM25 (Lexical) Results:
- '{entity_bn}' found {len(result_bm25_bn.results)} results (top score: {result_bm25_bn.top_score:.4f})
- '{entity_en}' found {len(result_bm25_en.results)} results (top score: {result_bm25_en.top_score:.4f})
- Overlap: {bm25_overlap}/5 documents

Semantic Results:
- '{entity_bn}' found {len(result_semantic_bn.results)} results (top score: {result_semantic_bn.top_score:.4f})
- '{entity_en}' found {len(result_semantic_en.results)} results (top score: {result_semantic_en.top_score:.4f})
- Overlap: {semantic_overlap}/5 documents

The semantic model {'successfully bridges' if success else 'fails to bridge'} the cross-lingual NER gap.
""",
            success=success,
            recommendations=[
                "Build a multilingual NER dictionary",
                "Use entity linking to normalize names",
                "Implement transliteration handling",
                "Add entity synonyms to search index"
            ]
        )
        
        self.case_studies.append(case)
        
        if verbose:
            print(case.to_markdown())
        
        return case
    
    def analyze_semantic_vs_lexical(
        self,
        query: str,
        expected_match_term: str,
        verbose: bool = True
    ) -> CaseStudy:
        """
        Analyze cases where semantic search wins over lexical
        
        Example: Query "শিক্ষা" (education), BM25 returns 0 results, 
                 but embedding model retrieves relevant "স্কুল" (school) documents
        
        Args:
            query: Original query (e.g., "শিক্ষা")
            expected_match_term: Related term that should match (e.g., "স্কুল")
            verbose: Print analysis
        """
        from module_d.ranking import RetrievalMethod
        
        if verbose:
            print("\n" + "=" * 70)
            print("🔍 SEMANTIC vs LEXICAL ANALYSIS")
            print("=" * 70)
        
        # BM25 search
        result_bm25 = self.ranking_system.search(
            query, method=RetrievalMethod.BM25, k=10, verbose=False
        )
        
        # Semantic search
        result_semantic = self.ranking_system.search(
            query, method=RetrievalMethod.SEMANTIC, k=10, verbose=False
        )
        
        # Check if semantic found documents containing related term
        semantic_titles = ' '.join([r.document.get('title', '') + r.document.get('body', '')[:200] 
                                    for r in result_semantic.results])
        found_related = expected_match_term.lower() in semantic_titles.lower()
        
        semantic_win = (len(result_semantic.results) > len(result_bm25.results) or
                       result_semantic.top_score > result_bm25.top_score)
        
        case = CaseStudy(
            category=ErrorCategory.SEMANTIC_WIN if semantic_win else ErrorCategory.LEXICAL_WIN,
            query=query,
            query_language="bn" if any('\u0980' <= c <= '\u09FF' for c in query) else "en",
            expected_behavior=f"Query '{query}' should find documents about related concept '{expected_match_term}'",
            actual_behavior=f"BM25: {len(result_bm25.results)} results (score: {result_bm25.top_score:.4f}), "
                           f"Semantic: {len(result_semantic.results)} results (score: {result_semantic.top_score:.4f})",
            retrieved_docs=[r.document | {"score": r.score} for r in result_semantic.results],
            analysis=f"""
Semantic vs Lexical Analysis:
- Query: '{query}'
- Related term: '{expected_match_term}'

BM25 (Lexical) Performance:
- Results found: {len(result_bm25.results)}
- Top score: {result_bm25.top_score:.4f}
- Requires exact word match

Semantic Performance:
- Results found: {len(result_semantic.results)}
- Top score: {result_semantic.top_score:.4f}
- Found related term '{expected_match_term}': {'Yes ✅' if found_related else 'No ❌'}

Winner: {'Semantic Search 🏆' if semantic_win else 'BM25 (Lexical) 🏆'}

This demonstrates {'the power of semantic understanding' if semantic_win else 'that exact matching can be more precise for specific terms'}.
""",
            success=True,  # This is analysis, not a failure
            recommendations=[
                "Use hybrid approach to get benefits of both",
                "Consider query expansion for BM25",
                "Fine-tune semantic model on domain-specific data",
                "Adjust hybrid weights based on query type"
            ]
        )
        
        self.case_studies.append(case)
        
        if verbose:
            print(case.to_markdown())
        
        return case
    
    def analyze_cross_script_ambiguity(
        self,
        term: str,
        transliterations: List[str],
        verbose: bool = True
    ) -> CaseStudy:
        """
        Analyze cross-script ambiguity
        
        Example: "Bangladesh" could be transliterated as "বাংলাদেশ" or "Bangla Desh" (two words)
        
        Args:
            term: Original term (e.g., "Bangladesh")
            transliterations: List of possible transliterations
            verbose: Print analysis
        """
        from module_d.ranking import RetrievalMethod
        
        if verbose:
            print("\n" + "=" * 70)
            print("🔍 CROSS-SCRIPT AMBIGUITY ANALYSIS")
            print("=" * 70)
        
        all_results = {}
        all_doc_ids = set()
        
        # Search with original term
        result_original = self.ranking_system.search(
            term, method=RetrievalMethod.HYBRID, k=5, verbose=False
        )
        all_results[term] = result_original
        all_doc_ids.update(r.doc_id for r in result_original.results)
        
        # Search with each transliteration
        for trans in transliterations:
            result = self.ranking_system.search(
                trans, method=RetrievalMethod.HYBRID, k=5, verbose=False
            )
            all_results[trans] = result
            all_doc_ids.update(r.doc_id for r in result.results)
        
        # Calculate overlap
        doc_id_sets = [set(r.doc_id for r in res.results) for res in all_results.values()]
        common_docs = doc_id_sets[0]
        for s in doc_id_sets[1:]:
            common_docs &= s
        
        success = len(common_docs) >= 2  # At least 2 common documents
        
        analysis_text = f"""
Cross-Script Ambiguity Analysis:
- Original term: '{term}'
- Transliterations tested: {transliterations}

Results by variant:
"""
        for variant, result in all_results.items():
            analysis_text += f"- '{variant}': {len(result.results)} results (score: {result.top_score:.4f})\n"
        
        analysis_text += f"""
- Total unique documents found: {len(all_doc_ids)}
- Documents common to ALL variants: {len(common_docs)}

The system {'handles' if success else 'struggles with'} cross-script ambiguity.
"""
        
        case = CaseStudy(
            category=ErrorCategory.CROSS_SCRIPT_AMBIGUITY,
            query=f"{term} / {' / '.join(transliterations)}",
            query_language="mixed",
            expected_behavior=f"All variants ({term}, {', '.join(transliterations)}) should retrieve similar documents",
            actual_behavior=f"Found {len(common_docs)} common documents across all variants",
            retrieved_docs=[r.document | {"score": r.score} for r in result_original.results],
            analysis=analysis_text,
            success=success,
            recommendations=[
                "Build transliteration normalization table",
                "Use character-level models for script-agnostic matching",
                "Implement query expansion with transliteration variants",
                "Consider phonetic matching algorithms"
            ]
        )
        
        self.case_studies.append(case)
        
        if verbose:
            print(case.to_markdown())
        
        return case
    
    def analyze_code_switching(
        self,
        mixed_query: str,
        pure_bn_query: str,
        pure_en_query: str,
        verbose: bool = True
    ) -> CaseStudy:
        """
        Analyze code-switching (mixing Bangla and English in query)
        
        Example: Query mixes Bangla and English words
        
        Args:
            mixed_query: Query with mixed languages (e.g., "Bangladesh এর election")
            pure_bn_query: Pure Bangla version
            pure_en_query: Pure English version
            verbose: Print analysis
        """
        from module_d.ranking import RetrievalMethod
        
        if verbose:
            print("\n" + "=" * 70)
            print("🔍 CODE-SWITCHING ANALYSIS")
            print("=" * 70)
        
        # Search with all variants
        result_mixed = self.ranking_system.search(
            mixed_query, method=RetrievalMethod.HYBRID, k=5, verbose=False
        )
        result_bn = self.ranking_system.search(
            pure_bn_query, method=RetrievalMethod.HYBRID, k=5, verbose=False
        )
        result_en = self.ranking_system.search(
            pure_en_query, method=RetrievalMethod.HYBRID, k=5, verbose=False
        )
        
        # Calculate overlaps
        mixed_ids = set(r.doc_id for r in result_mixed.results)
        bn_ids = set(r.doc_id for r in result_bn.results)
        en_ids = set(r.doc_id for r in result_en.results)
        
        mixed_bn_overlap = len(mixed_ids & bn_ids)
        mixed_en_overlap = len(mixed_ids & en_ids)
        
        success = len(result_mixed.results) > 0 and result_mixed.top_score > 0.3
        
        case = CaseStudy(
            category=ErrorCategory.CODE_SWITCHING,
            query=mixed_query,
            query_language="mixed (bn+en)",
            expected_behavior=f"Mixed query '{mixed_query}' should retrieve relevant documents despite code-switching",
            actual_behavior=f"Mixed: {len(result_mixed.results)} results, "
                           f"Overlap with pure Bangla: {mixed_bn_overlap}/5, "
                           f"Overlap with pure English: {mixed_en_overlap}/5",
            retrieved_docs=[r.document | {"score": r.score} for r in result_mixed.results],
            analysis=f"""
Code-Switching Analysis:
- Mixed query: '{mixed_query}'
- Pure Bangla: '{pure_bn_query}'
- Pure English: '{pure_en_query}'

Results:
- Mixed query: {len(result_mixed.results)} results (score: {result_mixed.top_score:.4f})
- Pure Bangla: {len(result_bn.results)} results (score: {result_bn.top_score:.4f})
- Pure English: {len(result_en.results)} results (score: {result_en.top_score:.4f})

Overlap Analysis:
- Mixed ∩ Bangla: {mixed_bn_overlap}/5 common documents
- Mixed ∩ English: {mixed_en_overlap}/5 common documents

The system {'handles code-switching well' if success else 'struggles with code-switching'}.
""",
            success=success,
            recommendations=[
                "Use multilingual embeddings trained on code-switched text",
                "Implement language detection at word level",
                "Consider separate processing for each language component",
                "Build a code-switching aware tokenizer"
            ]
        )
        
        self.case_studies.append(case)
        
        if verbose:
            print(case.to_markdown())
        
        return case
    
    def run_comprehensive_analysis(self, verbose: bool = True) -> List[CaseStudy]:
        """
        Run comprehensive error analysis with predefined test cases
        
        Args:
            verbose: Print detailed analysis
        
        Returns:
            List of all case studies
        """
        print("\n" + "=" * 80)
        print("🔬 COMPREHENSIVE ERROR ANALYSIS")
        print("=" * 80)
        
        # 1. Translation Failure
        print("\n📌 Test 1: Translation Failure")
        self.analyze_translation_failure(
            query="চেয়ার",  # chair in Bangla
            mistranslation="Chairman",  # common mistranslation
            correct_translation="chair",
            verbose=verbose
        )
        
        # 2. NER Mismatch
        print("\n📌 Test 2: Named Entity Mismatch")
        self.analyze_ner_mismatch(
            entity_bn="ঢাকা",  # Dhaka in Bangla
            entity_en="Dhaka",
            verbose=verbose
        )
        
        # 3. Semantic vs Lexical
        print("\n📌 Test 3: Semantic vs Lexical")
        self.analyze_semantic_vs_lexical(
            query="শিক্ষা",  # education in Bangla
            expected_match_term="স্কুল",  # school in Bangla
            verbose=verbose
        )
        
        # 4. Cross-Script Ambiguity
        print("\n📌 Test 4: Cross-Script Ambiguity")
        self.analyze_cross_script_ambiguity(
            term="Bangladesh",
            transliterations=["বাংলাদেশ", "Bangla Desh", "বাঙলাদেশ"],
            verbose=verbose
        )
        
        # 5. Code-Switching
        print("\n📌 Test 5: Code-Switching")
        self.analyze_code_switching(
            mixed_query="Bangladesh এর election",
            pure_bn_query="বাংলাদেশের নির্বাচন",
            pure_en_query="Bangladesh election",
            verbose=verbose
        )
        
        return self.case_studies
    
    def generate_report(self, filename: str = "error_analysis_report.md") -> str:
        """
        Generate a comprehensive markdown report
        
        Args:
            filename: Output filename
        
        Returns:
            Path to generated report
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# CLIR System Error Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Summary
            total = len(self.case_studies)
            successes = sum(1 for c in self.case_studies if c.success)
            failures = total - successes
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Case Studies:** {total}\n")
            f.write(f"- **Successes:** {successes} ✅\n")
            f.write(f"- **Failures/Issues:** {failures} ❌\n\n")
            
            # Category breakdown
            f.write("### By Category\n\n")
            f.write("| Category | Count | Success Rate |\n")
            f.write("|----------|-------|-------------|\n")
            
            from collections import Counter
            category_counts = Counter(c.category for c in self.case_studies)
            for cat, count in category_counts.items():
                cat_successes = sum(1 for c in self.case_studies if c.category == cat and c.success)
                rate = (cat_successes / count * 100) if count > 0 else 0
                f.write(f"| {cat.value.replace('_', ' ').title()} | {count} | {rate:.0f}% |\n")
            
            f.write("\n---\n\n")
            
            # Detailed case studies
            f.write("## Detailed Case Studies\n\n")
            
            for i, case in enumerate(self.case_studies, 1):
                f.write(f"## Case Study {i}\n")
                f.write(case.to_markdown())
            
            # Recommendations summary
            f.write("\n## Consolidated Recommendations\n\n")
            all_recs = set()
            for case in self.case_studies:
                all_recs.update(case.recommendations)
            
            for rec in sorted(all_recs):
                f.write(f"- {rec}\n")
        
        print(f"📄 Report generated: {filepath}")
        return filepath
    
    def save_case_studies_json(self, filename: str = "case_studies.json"):
        """Save all case studies to JSON"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([c.to_dict() for c in self.case_studies], f, ensure_ascii=False, indent=2)
        
        print(f"💾 Case studies saved: {filepath}")


# Demo usage
if __name__ == "__main__":
    print("=" * 70)
    print("Module D - Error Analysis Demo")
    print("=" * 70)
    print("\nNote: This module requires a RankingSystem instance to run analysis.")
    print("Use: analyzer = ErrorAnalyzer(ranking_system)")
    print("     analyzer.run_comprehensive_analysis()")
