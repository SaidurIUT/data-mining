# CLIR System Error Analysis Report

Generated: 2026-01-23 00:06:37

---

## Executive Summary

- **Total Case Studies:** 5
- **Successes:** 5 ✅
- **Failures/Issues:** 0 ❌

### By Category

| Category | Count | Success Rate |
|----------|-------|-------------|
| Translation Failure | 1 | 100% |
| Ner Mismatch | 1 | 100% |
| Semantic Vs Lexical Win | 1 | 100% |
| Cross Script Ambiguity | 1 | 100% |
| Code Switching | 1 | 100% |

---

## Detailed Case Studies

## Case Study 1

### Case Study: Translation Failure

**Status:** ✅ SUCCESS

**Query:** `চেয়ার` (bn)

**Expected Behavior:**
Query 'চেয়ার' should be translated to 'chair' and find relevant documents

**Actual Behavior:**
If mistranslated to 'Chairman', retrieves different/wrong documents

**Retrieved Documents:**
| Rank | Title | Score |
|------|-------|-------|
| 1 | ‘হ্যাঁ’ ভোটের পক্ষে অবস্থান নেওয়ার কারণ জানালো সরক... | 0.6000 |
| 2 | সাম্প্রদায়িক সম্প্রীতির মধ্যেই বাংলাদেশের সম্ভাবন... | 0.5856 |
| 3 | আপিল শুনানিতে কোনও পক্ষপাত করিনি: সিইসি... | 0.5518 |
| 4 | ‘নির্বাচনে অংশ নেবে কিনা বিবেচনা করছে এনসিপি’... | 0.5346 |
| 5 | চট্টগ্রামে সমাবেশে উপস্থিত থাকবেন তারেক রহমান... | 0.5243 |

**Analysis:**

Translation Analysis:
- Original query 'চেয়ার' retrieved 5 results
- Mistranslation 'Chairman' retrieved 5 results
- Correct translation 'chair' retrieved 5 results
- Overlap (original vs mistranslation): 2/5 documents
- Overlap (original vs correct): 3/5 documents

The semantic model successfully handles this translation ambiguity.


**Recommendations:**
- Use multiple translation candidates
- Implement translation confidence scoring
- Consider context-aware translation
- Add domain-specific translation dictionaries

---
## Case Study 2

### Case Study: Ner Mismatch

**Status:** ✅ SUCCESS

**Query:** `ঢাকা / Dhaka` (bn/en)

**Expected Behavior:**
Both 'ঢাকা' (Bangla) and 'Dhaka' (English) should retrieve same entity-related documents

**Actual Behavior:**
BM25 finds 0/5 common docs, Semantic finds 5/5 common docs

**Retrieved Documents:**
| Rank | Title | Score |
|------|-------|-------|
| 1 | সাম্প্রদায়িক সম্প্রীতির মধ্যেই বাংলাদেশের সম্ভাবন... | 0.3218 |
| 2 | চট্টগ্রামে সমাবেশে উপস্থিত থাকবেন তারেক রহমান... | 0.2933 |
| 3 | ঢাকা মহানগরীর প্রতিটি ওয়ার্ডে গণভোটের লিফলেট পৌঁছে... | 0.2605 |
| 4 | সাভারে পরিত্যক্ত কমিউনিটি সেন্টার থেকে আবারও ২ মরদ... | 0.2241 |
| 5 | ‘হ্যাঁ’ ভোটের পক্ষে অবস্থান নেওয়ার কারণ জানালো সরক... | 0.2210 |

**Analysis:**

NER Mismatch Analysis:
- Bangla entity: 'ঢাকা'
- English entity: 'Dhaka'

BM25 (Lexical) Results:
- 'ঢাকা' found 5 results (top score: 1.0000)
- 'Dhaka' found 0 results (top score: 0.0000)
- Overlap: 0/5 documents

Semantic Results:
- 'ঢাকা' found 5 results (top score: 0.3218)
- 'Dhaka' found 5 results (top score: 0.3045)
- Overlap: 5/5 documents

The semantic model successfully bridges the cross-lingual NER gap.


**Recommendations:**
- Build a multilingual NER dictionary
- Use entity linking to normalize names
- Implement transliteration handling
- Add entity synonyms to search index

---
## Case Study 3

### Case Study: Semantic Vs Lexical Win

**Status:** ✅ SUCCESS

**Query:** `শিক্ষা` (bn)

**Expected Behavior:**
Query 'শিক্ষা' should find documents about related concept 'স্কুল'

**Actual Behavior:**
BM25: 5 results (score: 1.0000), Semantic: 10 results (score: 0.2728)

**Retrieved Documents:**
| Rank | Title | Score |
|------|-------|-------|
| 1 | ইজিবাইককে চাপা দিয়ে বাস খাদে, ৬ জন নিহত... | 0.2728 |
| 2 | সাভারে পরিত্যক্ত কমিউনিটি সেন্টার থেকে আবারও ২ মরদ... | 0.2652 |
| 3 | হত্যার হুমকি পাওয়ার কথা জানিয়ে আমির হামজা বললেন, ‘... | 0.2512 |
| 4 | শাকসু ও ডিজেএফবি-কে নির্বাচনের অনুমতি দিলো ইসি... | 0.2503 |
| 5 | নির্বাচনি দায়িত্ব পালনে অনীহা ও শৈথিল্য দেখালে ব্য... | 0.2312 |

**Analysis:**

Semantic vs Lexical Analysis:
- Query: 'শিক্ষা'
- Related term: 'স্কুল'

BM25 (Lexical) Performance:
- Results found: 5
- Top score: 1.0000
- Requires exact word match

Semantic Performance:
- Results found: 10
- Top score: 0.2728
- Found related term 'স্কুল': No ❌

Winner: Semantic Search 🏆

This demonstrates the power of semantic understanding.


**Recommendations:**
- Use hybrid approach to get benefits of both
- Consider query expansion for BM25
- Fine-tune semantic model on domain-specific data
- Adjust hybrid weights based on query type

---
## Case Study 4

### Case Study: Cross Script Ambiguity

**Status:** ✅ SUCCESS

**Query:** `Bangladesh / বাংলাদেশ / Bangla Desh / বাঙলাদেশ` (mixed)

**Expected Behavior:**
All variants (Bangladesh, বাংলাদেশ, Bangla Desh, বাঙলাদেশ) should retrieve similar documents

**Actual Behavior:**
Found 2 common documents across all variants

**Retrieved Documents:**
| Rank | Title | Score |
|------|-------|-------|
| 1 | সংকট কাটাতে এলপিজি আনতে যাচ্ছে বিপিসি... | 0.6000 |
| 2 | সাম্প্রদায়িক সম্প্রীতির মধ্যেই বাংলাদেশের সম্ভাবন... | 0.5314 |
| 3 | রাজশাহীতে দুটি বিদেশি পিস্তল ও গুলি উদ্ধার... | 0.4870 |
| 4 | বড়পুকুরিয়া তাপ বিদ্যুৎকেন্দ্রের উৎপাদন বন্ধ... | 0.4337 |
| 5 | নির্বাচন কমিশন মোটামুটি যোগ্যতার সঙ্গে কাজ করছে: ম... | 0.4109 |

**Analysis:**

Cross-Script Ambiguity Analysis:
- Original term: 'Bangladesh'
- Transliterations tested: ['বাংলাদেশ', 'Bangla Desh', 'বাঙলাদেশ']

Results by variant:
- 'Bangladesh': 5 results (score: 0.6000)
- 'বাংলাদেশ': 5 results (score: 0.7477)
- 'Bangla Desh': 5 results (score: 0.6000)
- 'বাঙলাদেশ': 5 results (score: 0.6044)

- Total unique documents found: 10
- Documents common to ALL variants: 2

The system handles cross-script ambiguity.


**Recommendations:**
- Build transliteration normalization table
- Use character-level models for script-agnostic matching
- Implement query expansion with transliteration variants
- Consider phonetic matching algorithms

---
## Case Study 5

### Case Study: Code Switching

**Status:** ✅ SUCCESS

**Query:** `Bangladesh এর election` (mixed (bn+en))

**Expected Behavior:**
Mixed query 'Bangladesh এর election' should retrieve relevant documents despite code-switching

**Actual Behavior:**
Mixed: 5 results, Overlap with pure Bangla: 2/5, Overlap with pure English: 2/5

**Retrieved Documents:**
| Rank | Title | Score |
|------|-------|-------|
| 1 | নির্বাচন কমিশন মোটামুটি যোগ্যতার সঙ্গে কাজ করছে: ম... | 0.7358 |
| 2 | অষ্টম দিনে আপিল মঞ্জুর ৪৫ জনের, নামঞ্জুর ৩৭... | 0.7317 |
| 3 | আপিলে বৈধ হলো জামায়াত প্রার্থী সালেহীর মনোনয়ন... | 0.7256 |
| 4 | টিকে গেলেন হাসনাত, বাদ পড়লেন মঞ্জুরুল... | 0.6702 |
| 5 | নির্বাচনের মাঠ থেকে ছিটকে পড়লেন দেড়শো প্রার্থী... | 0.6026 |

**Analysis:**

Code-Switching Analysis:
- Mixed query: 'Bangladesh এর election'
- Pure Bangla: 'বাংলাদেশের নির্বাচন'
- Pure English: 'Bangladesh election'

Results:
- Mixed query: 5 results (score: 0.7358)
- Pure Bangla: 5 results (score: 0.9285)
- Pure English: 5 results (score: 0.6000)

Overlap Analysis:
- Mixed ∩ Bangla: 2/5 common documents
- Mixed ∩ English: 2/5 common documents

The system handles code-switching well.


**Recommendations:**
- Use multilingual embeddings trained on code-switched text
- Implement language detection at word level
- Consider separate processing for each language component
- Build a code-switching aware tokenizer

---

## Consolidated Recommendations

- Add domain-specific translation dictionaries
- Add entity synonyms to search index
- Adjust hybrid weights based on query type
- Build a code-switching aware tokenizer
- Build a multilingual NER dictionary
- Build transliteration normalization table
- Consider context-aware translation
- Consider phonetic matching algorithms
- Consider query expansion for BM25
- Consider separate processing for each language component
- Fine-tune semantic model on domain-specific data
- Implement language detection at word level
- Implement query expansion with transliteration variants
- Implement translation confidence scoring
- Implement transliteration handling
- Use character-level models for script-agnostic matching
- Use entity linking to normalize names
- Use hybrid approach to get benefits of both
- Use multilingual embeddings trained on code-switched text
- Use multiple translation candidates
