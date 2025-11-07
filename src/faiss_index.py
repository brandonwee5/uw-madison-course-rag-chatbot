import numpy as np
import faiss

# --- Step 1: Data Parsing Function ---
def parse_embeddings_file(file_path):
    """Reads the embeddings.txt file and returns two lists: course names and embeddings."""
    with open(file_path, 'r') as f:
        content = f.read()

    course_blocks = content.strip().split('============================================================')
    
    course_embeddings = []
    course_metadata = []

    for block in course_blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        
        try:
            course_id = lines[0].split(':')[1].strip()
            course_name = lines[1].split(':')[1].strip()
        except IndexError:
            continue

        embedding_list = []
        for line in lines[3:]:
            clean_line = line.strip().replace('[', '').replace(']', '')
            parts = [p.strip() for p in clean_line.split() if p.strip()]
            try:
                embedding_list.extend([float(p) for p in parts])
            except ValueError:
                continue
        
        if embedding_list:
            course_embeddings.append(embedding_list)
            course_metadata.append({"id": course_id, "name": course_name})
            
    embeddings_array = np.array(course_embeddings).astype('float32')
    
    return embeddings_array, course_metadata

# --- Execute Parsing ---
embeddings_array, course_metadata = parse_embeddings_file("data/processed/embeddings.txt")

# --- FAISS Index Setup ---
d = embeddings_array.shape[1]  # Dimensionality
nlist = 10                     # Number of cells/clusters

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(embeddings_array)
index.add(embeddings_array)
index.nprobe = 10
index.make_direct_map()

# --- Reconstruct a vector for verification ---
vec0 = index.reconstruct(0)
print("First 10 dimensions of vector 0:", vec0[:10])
print("Total vectors in index:", index.ntotal)
print("Index trained:", index.is_trained)

# --- Example Query ---
k = 3  # number of nearest neighbors to return
query_vector = embeddings_array[1].reshape(1, -1)  # use first embedding as a query

D, I = index.search(query_vector, k)  # distances and indices
print("\nQuery results (top-k nearest vectors):")
for rank, idx in enumerate(I[0]):
    print(f"Rank {rank + 1}: {course_metadata[idx]['name']} (Distance: {D[0][rank]:.4f})")
