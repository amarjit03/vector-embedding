import faiss
import numpy as np
import os, json

INDEX_PATH = "vectordb/index.faiss"
META_PATH = "vectordb/metadata.json"

def store_embeddings(vectors, metas):
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    os.makedirs("vectordb", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metas, f)

def search(query_vector, k=5):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r") as f:
        metas = json.load(f)

    distances, indices = index.search(np.array([query_vector]), k)
    return [{"text": metas[i], "score": float(dist)} for i, dist in zip(indices[0], distances[0])]
