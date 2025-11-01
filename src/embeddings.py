import json
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load your JSON file
with open("chunks.json", "r") as f:
    data = json.load(f)

# Extract the "text" field from each course chunk
sentences = [chunk["text"] for chunk in data]

# Convert sentences to embeddings
embeddings = model.encode(sentences)

# Print each course with its embedding
for chunk, emb in zip(data, embeddings):
    print(f"ID: {chunk['id']}")
    print(f"Course: {chunk['metadata']['course_name']}")
    print(f"Embedding (vector length {len(emb)}):")
    print(emb)
    print("="*60)
with open("embeddings.txt", "w") as f:
    for chunk, emb in zip(data, embeddings):
        f.write(f"ID: {chunk['id']}\n")
        f.write(f"Course: {chunk['metadata']['course_name']}\n")
        f.write(f"Embedding (length {len(emb)}):\n")
        f.write(str(emb) + "\n")
        f.write("="*60 + "\n")
sentences = [chunk["text"] for chunk in data]
unique_sentences = set(sentences)
print("Unique sentences:", len(unique_sentences))