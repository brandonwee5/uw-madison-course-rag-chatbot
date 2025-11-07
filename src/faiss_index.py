import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Step 1: Data Parsing Function ---
def parse_embeddings_file(file_path):
    """Reads the embeddings.txt file and returns embeddings and metadata."""
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


# --- Step 2: Initialize Embedding Model ---
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded: all-MiniLM-L6-v2 (384 dimensions)\n")


# --- Step 3: Parse Course Embeddings ---
print("Loading course embeddings...")
embeddings_array, course_metadata = parse_embeddings_file("data/processed/embeddings.txt")
print(f"✓ Loaded {len(course_metadata)} courses")
print(f"✓ Embedding dimension: {embeddings_array.shape[1]}\n")


# --- Step 4: Create FAISS Index ---
print("Creating FAISS IndexFlatL2...")
d = embeddings_array.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings_array)
print(f"✓ Index created with {index.ntotal} vectors\n")


# --- Step 5: Query Embedding Function ---
def embed_query(query_text):
    """Convert text query to embedding vector."""
    embedding = embedding_model.encode(query_text, convert_to_numpy=True)
    return embedding.astype('float32')


# --- Step 6: Retrieval Function ---
def retrieve_courses(query_text, k=3):
    """
    Retrieve top-k most relevant courses for a given query.
    
    Args:
        query_text: User's question or search query
        k: Number of courses to retrieve
        
    Returns:
        List of tuples: (course_metadata, distance)
    """
    print(f"Query: '{query_text}'")
    print("-" * 60)
    
    # Embed the query
    query_embedding = embed_query(query_text)
    query_vector = query_embedding.reshape(1, -1)
    
    # Search FAISS index
    distances, indices = index.search(query_vector, k)
    
    # Prepare results
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        course = course_metadata[idx]
        results.append((course, dist))
        
        print(f"{rank}. {course['name']}")
        print(f"   Course ID: {course['id']}")
        print(f"   L2 Distance: {dist:.4f}")
        print()
    
    return results


# --- Step 7: Test the Retrieval Pipeline ---
print("=" * 60)
print("TESTING RETRIEVAL PIPELINE")
print("=" * 60)
print()

# Test Query 1: About data analysis
print("TEST 1: Data Analysis Query")
print("=" * 60)
retrieve_courses("I want to learn about data analysis and statistics", k=3)

print("\n" + "=" * 60)
print("TEST 2: Health/Medicine Query")
print("=" * 60)
retrieve_courses("courses about health economics and policy", k=3)

print("\n" + "=" * 60)
print("TEST 3: Engineering/Biology Query")
print("=" * 60)
retrieve_courses("biological systems and engineering", k=3)

print("\n" + "=" * 60)
print("TEST 4: Communication/Media Query")
print("=" * 60)
retrieve_courses("science communication storytelling media", k=3)

print("\n" + "=" * 60)
print("TEST 5: Economics Query")
print("=" * 60)
retrieve_courses("economics decision making analysis", k=3)


# --- Step 8: Interactive Search (Optional) ---
print("\n" + "=" * 60)
print("INTERACTIVE SEARCH MODE")
print("=" * 60)
print("Enter your search queries (or 'quit' to exit):\n")

while True:
    try:
        user_query = input("Search: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_query:
            continue
            
        print()
        retrieve_courses(user_query, k=3)
        print()
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}\n")