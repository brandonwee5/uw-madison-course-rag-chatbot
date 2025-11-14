import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # or use anthropic, huggingface, etc.

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
            # Try to extract GPA if it exists in the file
            gpa = None
            for line in lines[:5]:  # Check first few lines for GPA
                if 'gpa' in line.lower() or 'average' in line.lower():
                    try:
                        gpa = float(line.split(':')[1].strip())
                    except (IndexError, ValueError):
                        pass
        except IndexError:
            continue

        embedding_list = []
        for line in lines[3:]:
            # Skip lines that might contain GPA info
            if 'gpa' in line.lower() or 'average' in line.lower():
                continue
            clean_line = line.strip().replace('[', '').replace(']', '')
            parts = [p.strip() for p in clean_line.split() if p.strip()]
            try:
                embedding_list.extend([float(p) for p in parts])
            except ValueError:
                continue
        
        if embedding_list:
            course_embeddings.append(embedding_list)
            metadata = {"id": course_id, "name": course_name}
            if gpa is not None:
                metadata["gpa"] = gpa
            course_metadata.append(metadata)
            
    embeddings_array = np.array(course_embeddings).astype('float32')
    
    return embeddings_array, course_metadata


# --- Step 2: Initialize Embedding Model ---
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded: all-MiniLM-L6-v2 (384 dimensions)\n")


# --- Step 3: Initialize LLM Client (Ollama) ---
print("Initializing local LLM client (Ollama)...")
# Using Ollama with OpenAI-compatible API
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
print("✓ Local LLM client initialized")
print("✓ Using Ollama (make sure Ollama is running!)\n")


# --- Step 4: Parse Course Embeddings ---
print("Loading course embeddings...")
embeddings_array, course_metadata = parse_embeddings_file("data/processed/embeddings.txt")
print(f"✓ Loaded {len(course_metadata)} courses")
print(f"✓ Embedding dimension: {embeddings_array.shape[1]}\n")


# --- Step 4: Load GPA and Grade Distribution Data ---
def load_course_details(chunks_file_path):
    """Load GPA and grade distribution data from chunks.json."""
    try:
        import json
        with open(chunks_file_path, 'r') as f:
            chunks = json.load(f)
        
        course_details = {}
        for chunk in chunks:
            course_id = chunk['id']
            metadata = chunk.get('metadata', {})
            course_details[course_id] = {
                'gpa': metadata.get('avg_gpa'),
                'grade_a': metadata.get('grade_a'),
                'grade_ab': metadata.get('grade_ab'),
                'grade_b': metadata.get('grade_b'),
                'grade_bc': metadata.get('grade_bc'),
                'grade_c': metadata.get('grade_c'),
                'grade_d': metadata.get('grade_d'),
                'grade_f': metadata.get('grade_f')
            }
        return course_details
    except FileNotFoundError:
        print(f"⚠ Chunks file not found: {chunks_file_path}")
        print("⚠ Running without GPA data.\n")
        return {}
    except Exception as e:
        print(f"⚠ Error loading course details: {e}\n")
        return {}

print("Loading course details (GPA, grades)...")
course_details = load_course_details("data/processed/chunks.json")

# Merge course details with metadata
for course in course_metadata:
    course_id = course['id']
    if course_id in course_details:
        details = course_details[course_id]
        course['gpa'] = details['gpa']
        course['grades'] = {
            'A': details['grade_a'],
            'AB': details['grade_ab'],
            'B': details['grade_b'],
            'BC': details['grade_bc'],
        }



if course_details:
    print(f"✓ Loaded details for {len(course_details)} courses\n")


# --- Step 5: Create FAISS Index ---
print("Creating FAISS IndexFlatL2...")
d = embeddings_array.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings_array)
print(f"✓ Index created with {index.ntotal} vectors\n")


# --- Step 6: Query Embedding Function ---
def embed_query(query_text):
    """Convert text query to embedding vector."""
    embedding = embedding_model.encode(query_text, convert_to_numpy=True)
    return embedding.astype('float32')


# --- Step 7: Retrieval Function ---
def retrieve_courses(query_text, k=3, verbose=True):
    """
    Retrieve top-k most relevant courses for a given query.
    
    Args:
        query_text: User's question or search query
        k: Number of courses to retrieve
        verbose: Whether to print results
        
    Returns:
        List of tuples: (course_metadata, distance)
    """
    if verbose:
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
        
        if verbose:
            print(f"{rank}. {course['name']}")
            print(f"   Course ID: {course['id']}")
            if 'gpa' in course:
                print(f"   Average GPA: {course['gpa']}")
            print(f"   L2 Distance: {dist:.4f}")
            print()
    
    return results


# --- Step 8: RAG Generation Function ---
def generate_rag_response(query_text, k=3, model="llama3.2"):
    """
    Complete RAG pipeline: Retrieve relevant courses and generate response.
    
    Args:
        query_text: User's question
        k: Number of courses to retrieve
        model: Ollama model to use (default: llama3.2)
               Other options: mistral, phi3, gemma2, etc.
        
    Returns:
        Generated response string
    """
    print(f"\n🔍 Retrieving relevant courses...")
    
    # Step 1: Retrieve relevant courses
    retrieved_courses = retrieve_courses(query_text, k=k, verbose=False)
    
    # Step 2: Format context from retrieved courses
    context = "Here are the most relevant courses:\n\n"
    for i, (course, dist) in enumerate(retrieved_courses, 1):
        context += f"{i}. Course: {course['name']}\n"
        context += f"   Course ID: {course['id']}\n"
        if 'gpa' in course and course['gpa']:
            context += f"   Average GPA: {course['gpa']:.3f}\n"
        if 'grades' in course:
            context += f"   Grade Distribution: A={course['grades']['A']}%, AB={course['grades']['AB']}%, B={course['grades']['B']}%\n"
        context += f"   Relevance Score: {1/(1+dist):.3f}\n\n"
    
    # Step 3: Create prompt for LLM
    system_prompt = """You are a course data assistant. 
You have access to course metadata including average GPA and grade distributions (percentages of A, AB, B, BC, C, D, F). 
Answer user questions using only this data, and do not give personal opinions. 
Provide clear, concise answers. 
If a user asks about a course's GPA, answer with the exact average GPA. 
If a user asks about the percentage of a specific grade in a course, provide the exact percentage. 
Always refer to the courses by their name and course ID as in the retrieved data."""
    
    user_prompt = f"""User Query: {query_text}

{context}

Please provide a helpful response recommending the most suitable courses for this query."""
    
    print("🤖 Generating response with LLM...\n")
    
    # Step 4: Generate response using LLM (Ollama)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content
        
    except Exception as e:
        answer = f"Error generating response: {e}\n\nRetrieved courses:\n"
        for i, (course, _) in enumerate(retrieved_courses, 1):
            answer += f"{i}. {course['name']} (ID: {course['id']})\n"
    
    return answer


# --- Step 9: Test RAG Pipeline ---
'''print("=" * 60)
print("TESTING COMPLETE RAG PIPELINE")
print("=" * 60)

test_queries = [
    "I want to learn about data analysis and statistics",
    "courses about health economics and policy",
    "biological systems and engineering",
    "science communication storytelling media",
    "economics decision making analysis"
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'='*60}")
    print(f"TEST {i}: {query}")
    print('='*60)
    
    response = generate_rag_response(query, k=3)
    
    print("📋 RESPONSE:")
    print("-" * 60)
    print(response)
    print()
'''


# --- Step 10: Interactive RAG Mode ---
print("\n" + "=" * 60)
print("INTERACTIVE RAG MODE")
print("=" * 60)
print("Ask questions about courses (or 'quit' to exit):\n")

while True:
    try:
        user_query = input("Your question: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not user_query:
            continue
        
        print()
        response = generate_rag_response(user_query, k=3)
        
        print("📋 RESPONSE:")
        print("-" * 60)
        print(response)
        print("\n" + "=" * 60 + "\n")
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}\n")