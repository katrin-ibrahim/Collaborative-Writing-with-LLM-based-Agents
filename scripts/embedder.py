from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os

from data_loader import load_wildseek

docs = load_wildseek() 

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)
embeddings = embeddings.astype(np.float32)

d = embeddings.shape[1]            
index = faiss.IndexFlatIP(d)       
faiss.normalize_L2(embeddings)
index.add(embeddings)              # now index.ntotal == len(docs)

os.makedirs("data/faiss", exist_ok=True)
faiss.write_index(index, "data/faiss/wildseek.index")

with open("data/faiss/wildseek_texts.pkl", "wb") as f:
    pickle.dump(docs, f)

print(f"Built index with {index.ntotal} vectors and saved to disk.")


