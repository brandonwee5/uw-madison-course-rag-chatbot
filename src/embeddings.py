import json
import numpy as np
from sentence_transformers import SentenceTransformer

def create_embeddings(chunks_file="../data/processed/chunks.json", output_file="embeddings.txt"):
    """
    Create embeddings from course chunks using sentence-transformers.
    
    Args:
        chunks_file: Path to chunks.json file
        output_file: Path to save embeddings
    """
    
    print("=" * 80)
    print("Creating Course Embeddings")
    print("=" * 80)
    
    # Load the embedding model
    print("\n📦 Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Model loaded: all-MiniLM-L6-v2 (384 dimensions)")
    
    # Load chunks
    print(f"\n📖 Loading chunks from {chunks_file}...")
    with open(chunks_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} course chunks")
    
    # Check for duplicates
    sentences = [chunk["text"] for chunk in data]
    unique_sentences = set(sentences)
    print(f"✅ Unique courses: {len(unique_sentences)}")
    
    if len(sentences) != len(unique_sentences):
        print(f"⚠️  Warning: Found {len(sentences) - len(unique_sentences)} duplicate courses")
    
    # Generate embeddings
    print("\n🔄 Generating embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"✅ Embedding shape: {embeddings.shape}")
    
    # Save embeddings to text file
    print(f"\n💾 Saving embeddings to {output_file}...")
    with open(output_file, "w", encoding='utf-8') as f:
        for chunk, emb in zip(data, embeddings):
            f.write(f"ID: {chunk['id']}\n")
            f.write(f"Course: {chunk['metadata']['course_name']}\n")
            f.write(f"Embedding (length {len(emb)}):\n")
            f.write(str(emb) + "\n")
            f.write("=" * 60 + "\n")
    
    print(f"✅ Embeddings saved to {output_file}")
    
    # Save embeddings in numpy format (more efficient for loading later)
    numpy_file = output_file.replace('.txt', '.npy')
    np.save(numpy_file, embeddings)
    print(f"✅ Embeddings also saved to {numpy_file} (numpy format)")
    
    # Show sample
    print("\n📊 Sample embedding:")
    print(f"Course: {data[0]['metadata']['course_name']}")
    print(f"First 10 dimensions: {embeddings[0][:10]}")
    
    return embeddings, data


def verify_embeddings(embeddings, data):
    """Verify embeddings were created correctly."""
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)
    
    # Check dimensions
    print(f"\n✓ Total embeddings: {len(embeddings)}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    print(f"✓ Expected dimension: 384")
    
    if embeddings.shape[1] != 384:
        print("❌ Error: Unexpected embedding dimension!")
        return False
    
    # Check for NaN or Inf values
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    
    if has_nan:
        print("❌ Error: Found NaN values in embeddings!")
        return False
    
    if has_inf:
        print("❌ Error: Found Inf values in embeddings!")
        return False
    
    print("✓ No NaN or Inf values found")
    
    # Check embedding statistics
    print(f"\n📈 Embedding Statistics:")
    print(f"   Mean: {np.mean(embeddings):.6f}")
    print(f"   Std:  {np.std(embeddings):.6f}")
    print(f"   Min:  {np.min(embeddings):.6f}")
    print(f"   Max:  {np.max(embeddings):.6f}")
    
    # Show some sample courses
    print(f"\n📚 Sample courses with embeddings:")
    for i in range(min(3, len(data))):
        course_name = data[i]['metadata']['course_name']
        gpa = data[i]['metadata']['avg_gpa']
        emb_norm = np.linalg.norm(embeddings[i])
        print(f"   {i+1}. {course_name}")
        print(f"      GPA: {gpa:.3f} | Embedding norm: {emb_norm:.4f}")
    
    print("\n✅ All verifications passed!")
    return True


if __name__ == "__main__":
    # Create embeddings
    embeddings, data = create_embeddings(
        chunks_file="/Users/bwee/uw-madison-course-rag-chatbot/data/processed/chunks.json",
        output_file="embeddings.txt"
    )
    
    # Verify embeddings
    verify_embeddings(embeddings, data)
    
    print("\n" + "=" * 80)
    print("✅ Embedding creation complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use 'embeddings.npy' for fast loading in your RAG system")
    print("2. Use 'embeddings.txt' for human inspection")
    print("3. Run your RAG chatbot with these embeddings!")