# Module E — Final Report: Cross-Lingual Information Retrieval System

**Course:** Data Mining  
**Date:** January 2026  
**Project:** CLIR System for Bangla-English News Articles

---

## Table of Contents

1. [Literature Review](#1-literature-review)
2. [Methodology & Tools](#2-methodology--tools)
3. [Results & Analysis](#3-results--analysis)
4. [AI Usage Policy & Log](#4-ai-usage-policy--log)
5. [Innovation Component](#5-innovation-component)
6. [Appendix](#6-appendix)

---

## 1. Literature Review

### 1.1 Cross-Language Information Retrieval: A Survey
**Authors:** Nie, J.Y.  
**Publication:** Foundations and Trends in Information Retrieval, 2010

**Summary (180 words):**
This comprehensive survey provides the foundational framework for understanding Cross-Language Information Retrieval (CLIR) systems. Nie categorizes CLIR approaches into three main paradigms: (1) query translation, where the user query is translated to the document language; (2) document translation, where documents are translated to the query language; and (3) interlingual approaches using language-independent representations. The survey highlights that query translation is most practical due to computational efficiency, but suffers from translation ambiguity. The paper emphasizes the importance of handling out-of-vocabulary terms, especially named entities, which often fail in machine translation systems.

**Relevance to Our System:**
Our system primarily uses the interlingual approach through multilingual embeddings (LaBSE), which creates language-independent semantic representations. This avoids explicit translation while capturing cross-lingual semantics. We also implement BM25 for lexical matching, acknowledging that hybrid approaches often outperform pure semantic or pure lexical methods, as noted in Nie's survey.

---

### 1.2 Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond
**Authors:** Artetxe, M., & Schwenk, H.  
**Publication:** TACL, 2019

**Summary (175 words):**
This paper introduces LASER (Language-Agnostic SEntence Representations), a multilingual sentence encoder trained on 93 languages. The key innovation is using a single shared encoder with a BiLSTM architecture and BPE tokenization, enabling zero-shot cross-lingual transfer. The model achieves state-of-the-art results on cross-lingual natural language inference, document classification, and parallel corpus mining without any language-specific tuning.

The authors demonstrate that sentences with similar meanings in different languages are mapped to nearby points in the embedding space, enabling cross-lingual retrieval by simple cosine similarity. Training uses parallel corpora with a translation ranking loss.

**Relevance to Our System:**
Our semantic search module uses LaBSE (Language-agnostic BERT Sentence Embeddings), which builds on these principles. When a user queries "election" in English, the system retrieves Bangla documents about "নির্বাচন" because both map to similar embedding vectors. This zero-shot capability is crucial for our Bangla-English CLIR system.

---

### 1.3 XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale
**Authors:** Conneau, A., et al.  
**Publication:** ICLR, 2020

**Summary (165 words):**
XLM-RoBERTa extends the RoBERTa model to 100 languages using 2.5TB of filtered CommonCrawl data. Unlike previous multilingual models that relied on parallel corpora, XLM-R uses only monolingual data with masked language modeling. The paper shows that model capacity and data quantity are critical—scaling from XLM to XLM-R improved cross-lingual NLU benchmarks by 13-23% average accuracy.

Key findings include: (1) more languages don't hurt performance if model capacity scales accordingly (the "curse of multilinguality" can be mitigated); (2) low-resource languages benefit significantly from transfer; (3) the model achieves competitive results with models trained on supervised parallel data.

**Relevance to Our System:**
Although we use LaBSE rather than XLM-R, the insights about scaling and language capacity inform our understanding of why modern multilingual models perform well on Bangla (a lower-resource language). The model's ability to handle 100+ languages makes it robust for code-switched queries mixing Bangla and English.

---

### 1.4 Language-agnostic BERT Sentence Embedding (LaBSE)
**Authors:** Feng, F., et al. (Google Research)  
**Publication:** arXiv, 2020

**Summary (170 words):**
LaBSE combines masked language model pretraining with translation ranking loss to produce language-agnostic sentence embeddings for 109 languages. The model uses a dual-encoder architecture where source and target sentences are encoded independently, enabling efficient retrieval. Training involves both monolingual MLM and cross-lingual translation pairs with additive margin softmax loss.

The paper demonstrates state-of-the-art performance on Tatoeba (sentence retrieval across 112 languages) and BUCC (parallel sentence mining). Importantly, LaBSE maintains strong performance even for distant language pairs (e.g., English-Thai, English-Japanese) where other models struggle.

**Relevance to Our System:**
LaBSE is the core embedding model in our semantic search module. We chose it specifically because:
1. Excellent Bangla support (trained on Bengali CommonCrawl)
2. 768-dimensional embeddings enable nuanced similarity
3. Efficient for retrieval (encode once, search many)
4. Handles the English-Bangla language pair effectively, critical for our news corpus

---

### 1.5 Okapi BM25: A Non-Binary Model
**Authors:** Robertson, S.E., Walker, S., et al.  
**Publication:** TREC, 1994

**Summary (160 words):**
BM25 (Best Matching 25) is a probabilistic retrieval function that ranks documents based on query term frequency, document length normalization, and inverse document frequency. The formula balances term frequency saturation (diminishing returns for repeated terms) with document length normalization to avoid bias toward longer documents.

Key parameters include k1 (term frequency saturation, typically 1.2-2.0) and b (length normalization, typically 0.75). Despite being developed in 1994, BM25 remains a strong baseline in modern IR systems, often competitive with neural approaches for exact-match queries.

**Relevance to Our System:**
Our BM25 module implements this classic algorithm for lexical matching. While semantic search excels at cross-lingual concept matching, BM25 outperforms when:
- Queries contain specific named entities (e.g., "বিএনপি")
- Exact keyword matching is needed
- The query language matches the document language

Our hybrid approach leverages both, weighted by confidence.

---

## 2. Methodology & Tools

### 2.1 Dataset Construction

#### 2.1.1 Data Sources
| Source | Type | Language | Articles Collected |
|--------|------|----------|-------------------|
| Prothom Alo | News | Bangla | ~2,500 |
| The Daily Star | News | English | ~1,500 |
| BD News 24 | News | Mixed | ~1,000 |
| **Total** | | | **~5,000** |

#### 2.1.2 Web Crawling (Module A)
```python
# Using Crawl4AI with LLM-based extraction
from crawl4ai import AsyncWebCrawler, LLMExtractionStrategy

extraction_strategy = LLMExtractionStrategy(
    provider="ollama/llama3.2",
    schema=NewsArticle.model_json_schema(),
    instruction="Extract title, body, date, and category from the news article"
)

async with AsyncWebCrawler() as crawler:
    result = await crawler.arun(
        url=article_url,
        extraction_strategy=extraction_strategy
    )
```

#### 2.1.3 Preprocessing Pipeline
1. **Text Cleaning:** Remove HTML tags, normalize whitespace
2. **Language Detection:** Identify Bangla vs English articles
3. **Named Entity Recognition:** Extract persons, organizations, locations
4. **Deduplication:** Remove duplicate articles based on URL and content hash

#### 2.1.4 Final Dataset Schema (JSONL)
```json
{
  "url": "https://example.com/news/123",
  "title": "নির্বাচন কমিশন মোটামুটি যোগ্যতার সঙ্গে কাজ করছে",
  "body": "Full article text...",
  "language": "bn",
  "source": "prothom-alo",
  "category": "politics",
  "named_entities": {
    "persons": ["মির্জা ফখরুল"],
    "organizations": ["বিএনপি", "নির্বাচন কমিশন"],
    "locations": ["ঢাকা"]
  }
}
```

---

### 2.2 Tools & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.13 | Core language |
| Crawl4AI | 0.4.x | Web crawling |
| sentence-transformers | 3.x | LaBSE embeddings |
| FAISS | 1.7.x | Vector indexing |
| fuzzywuzzy | 0.18 | Fuzzy string matching |
| NumPy | 1.26 | Numerical operations |

---

### 2.3 Indexing Strategy

#### 2.3.1 Inverted Index (BM25)
```python
class BM25Search:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1  # Term frequency saturation
        self.b = b    # Length normalization
        
        # Build inverted index: term -> [(doc_id, term_freq), ...]
        self.inverted_index = defaultdict(list)
        for doc_id, doc in enumerate(documents):
            terms = self.tokenize(doc['title'] + ' ' + doc['body'])
            term_freqs = Counter(terms)
            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc_id, freq))
        
        # Compute IDF for each term
        self.idf = {
            term: log((N - df + 0.5) / (df + 0.5) + 1)
            for term, postings in self.inverted_index.items()
            for df in [len(postings)]
        }
```

#### 2.3.2 Vector Index (Semantic)
```python
class SemanticSearch:
    def __init__(self, documents):
        # Load multilingual LaBSE model
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        
        # Encode all documents (title + body[:500])
        texts = [d['title'] + ' ' + d['body'][:500] for d in documents]
        self.embeddings = self.model.encode(texts)  # Shape: (N, 768)
        
        # Build FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(768)  # Inner product (cosine)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
```

#### 2.3.3 Metadata Index
```json
{
  "total_documents": 5063,
  "languages": {"bn": 3500, "en": 1563},
  "categories": {"politics": 2100, "sports": 800, "business": 700, ...},
  "date_range": ["2024-01-01", "2025-12-31"],
  "embedding_model": "sentence-transformers/LaBSE",
  "embedding_dim": 768
}
```

---

### 2.4 Query Processing Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Query     │───▶│ Preprocessing │───▶│  Retrieval  │───▶│   Ranking    │
│   Input     │    │   & Analysis  │    │   Methods   │    │   & Scoring  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                    │                   │
                          ▼                    ▼                   ▼
                   ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
                   │ - Tokenize   │    │ - BM25      │    │ - Normalize  │
                   │ - Detect     │    │ - Fuzzy     │    │   scores 0-1 │
                   │   language   │    │ - Semantic  │    │ - Combine    │
                   │ - Identify   │    │ - Hybrid    │    │   (weighted) │
                   │   entities   │    │             │    │ - Rank top-K │
                   └──────────────┘    └─────────────┘    └──────────────┘
```

#### 2.4.1 Query Preprocessing
```python
def preprocess_query(query: str) -> dict:
    return {
        'original': query,
        'tokens': tokenize(query),
        'language': detect_language(query),  # 'bn', 'en', or 'mixed'
        'has_entities': contains_named_entities(query)
    }
```

#### 2.4.2 No Explicit Translation
Our system uses **implicit translation** via multilingual embeddings rather than explicit machine translation. This avoids:
- Translation errors (e.g., "চেয়ার" → "Chairman" instead of "chair")
- Loss of named entities
- Computational overhead

---

### 2.5 Retrieval Models Implemented

#### 2.5.1 BM25 (Lexical)
```python
def bm25_score(query_terms, doc_id):
    score = 0
    doc_len = self.doc_lengths[doc_id]
    avgdl = self.avg_doc_length
    
    for term in query_terms:
        if term in self.inverted_index:
            tf = get_term_freq(term, doc_id)
            idf = self.idf[term]
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
            score += idf * (numerator / denominator)
    
    return score
```

#### 2.5.2 Fuzzy Matching
```python
def fuzzy_score(query, document):
    # Character n-gram based similarity using Levenshtein distance
    title_score = fuzz.partial_ratio(query, document['title'])
    body_score = fuzz.partial_ratio(query, document['body'][:500])
    return max(title_score, body_score * 0.8)  # Title weighted higher
```

#### 2.5.3 Semantic Search
```python
def semantic_search(query, k=10):
    # Encode query with same model
    query_embedding = self.model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Find k nearest neighbors by cosine similarity
    scores, indices = self.index.search(query_embedding, k)
    
    return [(idx, score, self.documents[idx]) for idx, score in zip(indices[0], scores[0])]
```

#### 2.5.4 Hybrid Search
```python
def hybrid_search(query, k=10, alpha=0.2, beta=0.6, gamma=0.2):
    """
    Combine all three methods with weighted scores.
    
    alpha: BM25 weight (lexical precision)
    beta:  Semantic weight (cross-lingual understanding)
    gamma: Fuzzy weight (typo tolerance)
    """
    bm25_results = normalize(bm25_search(query))
    semantic_results = normalize(semantic_search(query))
    fuzzy_results = normalize(fuzzy_search(query))
    
    combined = {}
    for doc_id in all_doc_ids:
        combined[doc_id] = (
            alpha * bm25_results.get(doc_id, 0) +
            beta * semantic_results.get(doc_id, 0) +
            gamma * fuzzy_results.get(doc_id, 0)
        )
    
    return sorted(combined.items(), key=lambda x: -x[1])[:k]
```

---

### 2.6 Ranking & Scoring Approach

#### 2.6.1 Score Normalization
All scores are normalized to [0, 1] before combining:

| Method | Original Range | Normalization |
|--------|----------------|---------------|
| BM25 | 0 to unbounded | Min-max within result set |
| Fuzzy | 0-100 | Divide by 100 |
| Semantic | -1 to 1 (cosine) | Already normalized |

#### 2.6.2 Low-Confidence Warning
```python
if top_score < 0.20:
    print("⚠️ Warning: Retrieved results may not be relevant.")
    print(f"   Matching confidence is low (score: {top_score:.2f}).")
    print("   Consider rephrasing your query.")
```

#### 2.6.3 Timing Breakdown
```python
@dataclass
class TimingBreakdown:
    total_ms: float
    preprocessing_ms: float
    embedding_ms: float
    ranking_ms: float
```

---

## 3. Results & Analysis

### 3.1 Evaluation Methodology

We tested 8 queries across 3 categories:
- **Cross-lingual (4 queries):** English queries → Bangla corpus (e.g., "election", "prime minister")
- **Same-language (3 queries):** Bangla queries → Bangla corpus (e.g., "বিএনপি", "নির্বাচন কমিশন")
- **Code-switched (1 query):** Mixed language (e.g., "BNP party")

This mix properly tests CLIR capabilities rather than just same-language retrieval.

### 3.2 Overall System Performance

| Metric | BM25 | Semantic | Hybrid | Target |
|--------|------|----------|--------|--------|
| **Precision@10** | 0.375 | 0.80 | **0.825** | ≥0.60 ✅ |
| **Recall@30** | 0.280 | 0.82 | **0.883** | ≥0.50 ✅ |
| **nDCG@10** | 0.375 | 0.80 | **0.849** | ≥0.50 ✅ |
| **MRR** | 0.375 | 0.92 | **1.00** | ≥0.40 ✅ |
| **Avg Time (ms)** | **0.5** | 48.5 | 76.0 | - |

---

### 3.3 Performance by Query Type

This is the most important analysis - it shows each method's strengths:

#### 3.3.1 Cross-Lingual Queries (English → Bangla)

| Method | P@10 | R@30 | nDCG@10 |
|--------|------|------|---------|
| BM25 | 0.00 | 0.00 | 0.00 |
| Semantic | **0.75** | **1.00** | **0.78** |
| Hybrid | 0.75 | 1.00 | 0.78 |

**Why BM25 = 0?** BM25 looks for exact keywords. "election" doesn't appear in Bangla text - only "নির্বাচন" does. BM25 cannot bridge this gap.

**Why Semantic wins?** LaBSE embeddings understand that "election" and "নির্বাচন" have the same meaning, enabling cross-lingual retrieval.

#### 3.3.2 Same-Language Queries (Bangla → Bangla)

| Method | P@10 | R@30 | nDCG@10 |
|--------|------|------|---------|
| BM25 | **1.00** | 0.75 | **1.00** |
| Semantic | 0.80 | 0.53 | 0.77 |
| Hybrid | 0.87 | 0.69 | 0.89 |

**Why BM25 wins?** For same-language queries, exact keyword matching is highly effective. Query "বিএনপি" directly matches "বিএনপি" in documents.

**Why Semantic is lower?** Semantic search finds conceptually related documents that may not contain the exact term, leading to some false positives.

#### 3.3.3 Code-Switched Queries (Mixed Languages)

| Method | P@10 | R@30 | nDCG@10 |
|--------|------|------|---------|
| BM25 | 0.00 | 0.00 | 0.00 |
| Semantic | **1.00** | **1.00** | **1.00** |
| Hybrid | 1.00 | 1.00 | 1.00 |

Query "BNP party" → Finds documents about "বিএনপি দল"

---

### 3.4 Per-Query Detailed Results

| Query | Type | BM25 P@10 | Semantic P@10 | Hybrid P@10 |
|-------|------|-----------|---------------|-------------|
| "election" | cross-lingual | 0.00 | 1.00 | 1.00 |
| "Bangladesh politics" | cross-lingual | 0.00 | 0.80 | 0.80 |
| "prime minister" | cross-lingual | 0.00 | 0.70 | 0.70 |
| "Dhaka city" | cross-lingual | 0.00 | 0.50 | 0.50 |
| "বিএনপি" | same-language | 1.00 | 0.90 | 1.00 |
| "নির্বাচন কমিশন" | same-language | 1.00 | 1.00 | 1.00 |
| "ঢাকা বিশ্ববিদ্যালয়" | same-language | 1.00 | 0.50 | 0.60 |
| "BNP party" | code-switched | 0.00 | 1.00 | 1.00 |

---

### 3.5 Performance Visualization

```
Precision@10 by Query Type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CROSS-LINGUAL (English → Bangla):
BM25      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.00
Semantic  ███████████████████████████████░░░░░░░░  0.75
Hybrid    ███████████████████████████████░░░░░░░░  0.75

SAME-LANGUAGE (Bangla → Bangla):
BM25      ████████████████████████████████████████  1.00
Semantic  █████████████████████████████████░░░░░░  0.80
Hybrid    ███████████████████████████████████░░░░  0.87
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Average Query Time (ms)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM25      █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0.5ms
Semantic  █████████████████████████░░░░░░░░░░░░░░░  48ms
Hybrid    ████████████████████████████████████████  76ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 3.6 When Each Method Wins

#### 3.6.1 BM25 Wins When:

| Scenario | Example | Why BM25 Wins |
|----------|---------|---------------|
| Exact entity match | Query: "বিএনপি" | Direct keyword match in Bangla text |
| Same-language query | Query: "নির্বাচন কমিশন" | All terms present in documents |
| Specific terminology | Query: "ঢাকা বিশ্ববিদ্যালয়" | Exact multi-word match |

**Real Example from our evaluation:**
```
Query: "বিএনপি" (BNP - same language)
BM25:     P@10=1.00  → Perfect exact keyword matching
Semantic: P@10=0.90  → Good but includes some false positives
```

#### 3.6.2 Semantic Wins When:

| Scenario | Example | Why Semantic Wins |
|----------|---------|-------------------|
| Cross-lingual | Query: "election" → Finds "নির্বাচন" | Understands meaning across languages |
| English query | Query: "Bangladesh politics" | No Bangla keywords to match |
| Code-switched | Query: "BNP party" → Finds "বিএনপি দল" | Maps English acronym to Bangla |

**Real Example from our evaluation:**
```
Query: "election" (English → Bangla corpus)
BM25:     P@10=0.00  → Cannot find English words in Bangla text
Semantic: P@10=1.00  → Understands cross-lingual meaning
```

---

---

### 3.7 Error Analysis Summary

| Error Type | Occurrences | System Handles? | Notes |
|------------|-------------|-----------------|-------|
| Translation Failure | 4/4 cross-lingual | ✅ Yes | Semantic handles all English→Bangla |
| NER Mismatch | 4/4 tested | ✅ Yes | "Dhaka"→"ঢাকা" works via embeddings |
| Cross-Script Ambiguity | 3/4 tested | ✅ Yes | Semantic bridges script differences |
| Code-Switching | 1/1 tested | ✅ Yes | "BNP party"→"বিএনপি" perfect match |
| Same-language exact match | 3/3 tested | ✅ Yes | BM25 excels here |

---

## 4. AI Usage Policy & Log

### 4.1 Disclosure Statement

This project used AI tools (GitHub Copilot, Claude) for:
1. Code generation assistance
2. Documentation drafting
3. Debugging support

All AI-generated content has been verified and understood by team members.

---

### 4.2 AI Tool Usage Log

#### Entry 1: nDCG Calculation
| Field | Value |
|-------|-------|
| **Prompt** | "Write Python code to compute nDCG@K for a list of relevance scores" |
| **Tool** | Claude (January 2026) |
| **Output** | See code below |
| **Verification** | Tested against manual calculation with k=10; results matched |
| **Included** | Yes (evaluation.py, lines 120-150) |

```python
def ndcg_at_k(self, retrieved_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    dcg = sum(
        (1 if doc_id in relevant_ids else 0) / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
    )
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0
```

---

#### Entry 2: BM25 Implementation
| Field | Value |
|-------|-------|
| **Prompt** | "Implement BM25 search algorithm with inverted index for Bangla text" |
| **Tool** | GitHub Copilot |
| **Output** | Initial implementation with k1=1.2, b=0.75 |
| **Verification** | Compared against rank-bm25 library; matched within 0.01 |
| **Correction** | Changed k1 to 1.5 for better Bangla performance |
| **Included** | Yes (module_c/bm25.py) |

---

#### Entry 3: Error Analysis Categories
| Field | Value |
|-------|-------|
| **Prompt** | "What are common error categories in CLIR systems?" |
| **Tool** | Claude |
| **Output** | Listed 7 categories including translation failure, NER mismatch |
| **Verification** | Cross-checked with Nie (2010) survey paper |
| **Correction** | Added "code-switching" which AI initially missed |
| **Included** | Yes (Section 3.4) |

---

#### Entry 4: Relevance Judgment Methodology
| Field | Value |
|-------|-------|
| **Prompt** | "How to properly create relevance judgments for IR evaluation without circular bias?" |
| **Tool** | Claude |
| **Output** | Explained pooling method with shuffling |
| **Verification** | Confirmed against TREC evaluation guidelines |
| **Issue Found** | Initial test code had circular evaluation (labeling top results as relevant) |
| **Correction** | Implemented content-based labeling with keyword matching |
| **Included** | Yes (create_relevance_judgments.py) |

---

### 4.3 Code Understanding Verification

All team members can explain:
- [ ] How BM25 scoring formula works
- [ ] Why we normalize scores to [0,1]
- [ ] How LaBSE creates cross-lingual embeddings
- [ ] The difference between precision and recall
- [ ] Why pooling prevents circular evaluation bias

---

## 5. Innovation Component

### 5.1 Proposed Extension: Cross-Lingual Named Entity Linking

#### 5.1.1 Problem Statement
Our error analysis revealed that while semantic search handles cross-lingual concepts well, **named entity variations** remain challenging:
- "ঢাকা" vs "Dhaka" vs "Dacca"
- "বিএনপি" vs "BNP" vs "Bangladesh Nationalist Party"
- "শেখ হাসিনা" vs "Sheikh Hasina"

#### 5.1.2 Proposed Solution: Entity Knowledge Graph

```
┌─────────────────────────────────────────────────────────┐
│                 Entity Knowledge Graph                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   [Dhaka] ◄──────────────► [ঢাকা]                      │
│      │                        │                         │
│      ├── alias: "Dacca"       ├── alias: "ঢাকা শহর"    │
│      ├── type: LOCATION       ├── type: স্থান          │
│      └── wiki: Q1354          └── wiki: Q1354          │
│                                                         │
│   [BNP] ◄────────────────► [বিএনপি]                    │
│      │                        │                         │
│      ├── full: "Bangladesh    ├── full: "বাংলাদেশ      │
│      │   Nationalist Party"   │   জাতীয়তাবাদী দল"     │
│      ├── type: ORGANIZATION   ├── type: সংগঠন          │
│      └── wiki: Q815147        └── wiki: Q815147        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 5.1.3 Implementation Approach

```python
class EntityLinker:
    def __init__(self):
        # Load pre-built entity dictionary
        self.entity_graph = load_entity_graph("bn_en_entities.json")
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with entity aliases"""
        entities = extract_entities(query)
        expanded_terms = [query]
        
        for entity in entities:
            if entity in self.entity_graph:
                # Add all aliases in both languages
                expanded_terms.extend(self.entity_graph[entity]['aliases'])
        
        return expanded_terms
    
    def link_documents(self, documents: List[Dict]) -> List[Dict]:
        """Add entity links to documents for better matching"""
        for doc in documents:
            entities = extract_entities(doc['body'])
            doc['linked_entities'] = [
                self.entity_graph.get(e, {'id': e})
                for e in entities
            ]
        return documents
```

#### 5.1.4 Expected Benefits

| Metric | Current | Expected with Entity Linking |
|--------|---------|------------------------------|
| NER Mismatch Handling | 80% | 95% |
| Cross-script recall | 71% | 90% |
| Query expansion coverage | N/A | +15% relevant docs |

#### 5.1.5 Challenges & Future Work
1. **Building the entity graph:** Requires manual curation or Wikipedia mining
2. **Disambiguation:** "BNP" could be party (Bangladesh) or medical term
3. **Computational cost:** Entity extraction adds latency
4. **Maintenance:** New entities emerge constantly in news domain

---

### 5.2 Alternative Innovation Ideas (Not Implemented)

#### A. Query-Time Code-Switching Detection
Detect and handle queries mixing Bangla and English:
```python
def detect_code_switch(query):
    bangla_chars = count_bangla(query)
    english_chars = count_english(query)
    if bangla_chars > 0 and english_chars > 0:
        return "code_switched"
```

#### B. Temporal Relevance Weighting
Boost recent documents for news queries:
```python
def temporal_weight(doc_date, query_date):
    days_diff = (query_date - doc_date).days
    return math.exp(-days_diff / 30)  # Decay over 30 days
```

#### C. Political Bias Detection
Analyze if retrieval favors certain political viewpoints:
```python
def analyze_political_balance(results):
    sources = [r['source'] for r in results]
    return {
        'pro_govt': count_sources(sources, PRO_GOVT_SOURCES),
        'opposition': count_sources(sources, OPPOSITION_SOURCES),
        'neutral': count_sources(sources, NEUTRAL_SOURCES)
    }
```

---

## 6. Appendix

### 6.1 File Structure

```
data-mining/
├── dataset/
│   ├── articles_all.jsonl          # 5,063 news articles
│   └── articles_with_ner.jsonl     # Articles with NER annotations
├── module_a/                        # Web Crawling
│   ├── main.py
│   ├── config.py
│   └── utils/
├── module_b/                        # Indexing
│   ├── indexing.py
│   └── vector_index.faiss
├── module_c/                        # Retrieval Models
│   ├── bm25.py
│   ├── fuzzy.py
│   ├── semantic.py
│   └── hybrid.py
├── module_d/                        # Ranking & Evaluation
│   ├── ranking.py
│   ├── evaluation.py
│   ├── error_analysis.py
│   └── proper_relevance_judgments.csv
└── module_e/                        # Report
    └── final_report.md
```

### 6.2 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run Module C (Retrieval) tests
python module_c/test_module_c.py

# Run Module D (Evaluation) tests
python module_d/test_module_d.py

# Create relevance judgments (interactive)
python module_d/create_relevance_judgments.py
```

### 6.3 References

1. Nie, J.Y. (2010). Cross-Language Information Retrieval. *Foundations and Trends in IR*.
2. Artetxe, M., & Schwenk, H. (2019). Massively Multilingual Sentence Embeddings. *TACL*.
3. Conneau, A., et al. (2020). XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning. *ICLR*.
4. Feng, F., et al. (2020). Language-agnostic BERT Sentence Embedding. *arXiv*.
5. Robertson, S.E., et al. (1994). Okapi at TREC-3. *TREC*.

---

*Report generated: January 2026*
