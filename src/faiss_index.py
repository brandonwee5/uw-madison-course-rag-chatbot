import numpy as np
import faiss

# --- Step 1: Data Parsing Function ---
def parse_embeddings_file(file_path):
    """Reads the embeddings.txt file and returns two lists: course names and embeddings."""
    
    with open(file_path, 'r') as f:
        content = f.read()

    # Split the content by the course separator (e.g., '================')
    course_blocks = content.strip().split('============================================================')
    
    course_embeddings = []
    course_metadata = []

    for block in course_blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        
        # Extract ID and Course Name
        try:
            course_id = lines[0].split(':')[1].strip()    # first line: "ID: course_0"
            course_name = lines[1].split(':')[1].strip()  # second line: "Course: Intro to Data Analysis"

            # The next line confirms length, e.g., 'Embedding (length 384):'
        except IndexError:
            # Skip blocks that don't conform to the expected header structure
            continue

        # Extract numerical embedding values from the remaining lines
        embedding_list = []
        for line in lines[3:]:
            # Clean up the line: remove brackets, replace 'e' with 'e', split by spaces
            clean_line = line.strip().replace('[', '').replace(']', '')
            # Filter out empty strings that result from multiple spaces
            parts = [p.strip() for p in clean_line.split() if p.strip()]
            
            try:
                # Convert cleaned string parts to floats
                embedding_list.extend([float(p) for p in parts])
            except ValueError:
                # Handle cases where non-float data sneaks in (like extra blank lines)
                continue
        
        if embedding_list:
            course_embeddings.append(embedding_list)
            course_metadata.append({
                "id": course_id,
                "name": course_name
            })
            
    # Convert the list of embeddings into a single NumPy array (required by FAISS)
    # Ensure float32 datatype for FAISS compatibility
    embeddings_array = np.array(course_embeddings).astype('float32')
    
    return embeddings_array, course_metadata

# --- Execute Parsing ---
# NOTE: Replace 'embeddings.txt' with your actual file path if needed
embeddings_array, course_metadata = parse_embeddings_file("data/processed/embeddings.txt")


# --- Verification ---
d = embeddings_array.shape[1] # Dimensionality
N = embeddings_array.shape # Number of vectors

#print(f"Total Vectors loaded (N): {N}")
#print(f"Vector Dimensionality (d): {d}") 
#print(f"Example Course Name: {course_metadata[0]['name']}")

nlist = 10  # how many cells
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(embeddings_array)  # pass the NumPy array, 

# Add embeddings to the index
index.add(embeddings_array)
index.nprobe = 10
index.make_direct_map()
vec0 = index.reconstruct(0)
m = 8  # number of centroid IDs in final compressed vectors
bits = 8 # number of bits in each centroid
index.is_trained
index.train(embeddings_array)
index.add(embeddings_array)
quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits) 
index.nprobe = 10
print("First 10 dimensions of vector 0:", vec0[:10])