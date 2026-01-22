"""
Module D - Ranking, Scoring & Evaluation
=========================================
This module provides ranking, scoring, evaluation metrics, and error analysis
for the CLIR (Cross-Lingual Information Retrieval) system.

Components:
- ranking.py: Ranking functions with normalized scores and confidence warnings
- evaluation.py: IR metrics (Precision, Recall, nDCG, MRR)
- error_analysis.py: Detailed error analysis for retrieval failures
- relevance_labels.py: Tools for creating relevance judgments

Usage:
    from module_d import RankingSystem, Evaluator, ErrorAnalyzer
"""

from module_d.ranking import RankingSystem
from module_d.evaluation import Evaluator
from module_d.error_analysis import ErrorAnalyzer

__all__ = ['RankingSystem', 'Evaluator', 'ErrorAnalyzer']
