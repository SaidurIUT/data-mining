import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load the Index and Metadata
index = faiss.read_index("vector_index.faiss")
with open("metadata.json", "r", encoding='utf-8') as f:
    meta = json.load(f)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. Define Entity Map (Task 5)
# In a real run, this would be mined from your articles_with_ner.jsonl
entity_map = {
    "bnp": "বিএনপি",
    "tarek": "তারেক রহমান",
    "nahid": "নাহিদ ইসলাম"
}

def search_test(user_query):
    print(f"\n--- Testing Query: '{user_query}' ---")
    
    # Task 4 & 5: Query Expansion / Entity Mapping
    processed_query = user_query.lower()
    for en, bn in entity_map.items():
        if en in processed_query:
            processed_query += f" {bn}"
            print(f"Mapped Entity: {en} -> {bn}")

    # Convert to Vector
    vec = model.encode([processed_query])
    faiss.normalize_L2(vec)

    # Search (k=1 means get the top result)
    distances, indices = index.search(vec, k=1)
    
    match_idx = indices[0][0]
    score = distances[0][0]
    
    print(f"Match Found: {meta[match_idx]['title']}")
    print(f"Similarity Score: {score:.4f}")
    print(f"Source: {meta[match_idx]['source']}")

# --- Run Dry Run Tests ---

# Test 1: Search for the Prothom Alo Opinion piece (Nahid Islam)
search_test("wealth of Nahid Islam")

# Test 2: Search for the weapons recovery article
search_test("police weapons recovered in Araihazar")

# Test 3: Search using an Entity Map trigger
search_test("latest news about Tarek Rahman")