import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss_index

with open("chunks.json", "r") as f:
    data = json.load(f)

sentences = [chunk["text"] for chunk in data]
sentences = [s for s in list(set(sentences)) if type(s) is str]

print("Unique sentences:", len(sentences))

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences)
sentence_embeddings = np.array(sentence_embeddings).astype('float32')

d = sentence_embeddings.shape[1]
index = faiss_index.IndexFlatL2(d)
index.add(sentence_embeddings)

print("Number of vectors in index:", index.ntotal)
print("Is FAISS index trained?", index.is_trained)
print("Embedding dimension:", d)

k = 4
xq = model.encode(["Intro to Data Analysis"])
xq = np.array(xq).astype('float32')

D, I = index.search(xq, k)

print("Nearest neighbor indices:", I)
print("Distances:", D)

# Optionally, map indices back to sentences
print("\nNearest neighbor sentences:")
for idx in I[0]:
    print(sentences[idx])
