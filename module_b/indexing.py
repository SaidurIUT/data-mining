import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

# --- TOGGLE THIS ---
DRY_RUN = True  # Set to True for 3 records, False for all 5000+
# -------------------

# 1. Configuration
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
INPUT_FILE = "/Users/kaziakibzaoad/Academic/7TH SEMESTER/Data Mining/Assignment/code/dataset/articles_all.jsonl"
INDEX_PATH = "vector_index.faiss"
METADATA_PATH = "metadata.json"

# 2. Load Model
print(f"Loading Model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

def build_index():
    documents = []
    metadata = []
    
    print(f"Reading JSONL (DRY_RUN is {DRY_RUN})...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # If DRY_RUN is True, stop after 3 records. 
            # If False, loop through all 5000+.
            if DRY_RUN and i >= 3:
                break
                
            item = json.loads(line)
            # Combine Title + Body
            full_text = f"{item.get('title', '')} {item.get('body', '')}"
            documents.append(full_text)
            
            # Keep metadata to display results later
            metadata.append({
                "id": i,
                "title": item.get("title"),
                "url": item.get("url"),
                "source": item.get("source")
            })

    print(f"Generating embeddings for {len(documents)} records...")
    # This encodes the text into vectors
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    
    # 3. Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    faiss.normalize_L2(embeddings)       
    index.add(embeddings)

    # 4. Save everything
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Done! Files saved: {INDEX_PATH}, {METADATA_PATH}")

if __name__ == "__main__":
    build_index()