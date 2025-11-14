"""
rag_pipeline_instrumented.py
Complete RAG pipeline with comprehensive timing metrics

Usage:
    # Run full pipeline (stages 1-3)
    python rag_pipeline_instrumented.py --build

    # Run chatbot only (stage 4)
    python rag_pipeline_instrumented.py --chat

    # View performance report
    python rag_pipeline_instrumented.py --report
"""

import time
import os
import json
import re
import sys
import numpy as np
from pathlib import Path

# Global stats storage
STATS_FILE = "pipeline_stats.json"
query_timings = []


# ============================================================================
# STAGE 1: PDF → MARKDOWN
# ============================================================================

def stage1_pdf_to_markdown(pdf_file="/data/report-gradedistribution-2024-2025spring.pdf", 
                           output_file="fullgrades.md"):
    """Stage 1: Convert PDF to Markdown using LlamaParse"""
    from llama_parse import LlamaParse
    from dotenv import load_dotenv
    
    print("=" * 80)
    print("STAGE 1: PDF → MARKDOWN")
    print("=" * 80)
    
    load_dotenv()
    
    input_size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
    print(f"\n📄 Input: {pdf_file} ({input_size_mb:.2f} MB)")
    
    parser = LlamaParse(result_type="markdown")
    
    print("🔄 Parsing PDF...")
    start_time = time.perf_counter()
    
    with open(pdf_file, "rb") as f:
        extra_info = {"file_name": os.path.basename(pdf_file)}
        documents = parser.load_data(f, extra_info=extra_info)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.text + "\n\n")
    
    wall_time = time.perf_counter() - start_time
    output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    stats = {
        "wall_time_seconds": round(wall_time, 3),
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "num_chunks": len(documents),
        "throughput_mb_per_sec": round(input_size_mb / wall_time, 3)
    }
    
    print(f"✅ Complete! Wall Time: {wall_time:.3f}s | Output: {output_size_mb:.2f} MB")
    return stats


# ============================================================================
# STAGE 2: MARKDOWN → CHUNKS
# ============================================================================

def parse_markdown_to_chunks(input_file="fullgrades.md"):
    """Parse markdown tables into course chunks"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = []
    lines = content.split('\n')
    current_course_name = None
    
    for line in lines:
        line = line.strip()
        
        if not line or '|' not in line:
            continue
        
        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]
        
        if len(parts) < 5:
            continue
        
        first_col = parts[0]
        
        # Skip headers and summaries
        skip_keywords = [
            'Section', 'Grades', 'Total', 'GPA', 'Ave', 'Avg', '#',
            'Percentage', 'Distribution', 'Office', 'Registrar',
            'University', 'Wisconsin', 'Madison', 'TERM', 'Page',
            'Freshmen', 'Sophomore', 'Junior', 'Senior',
            'Graduate', 'Special', 'Undergraduates', 'Professionals',
            'Summary by Level', 'Dept. Total', 'Course Total',
            'Prof Yr', 'Please note', 'ALS', 'BSE', 'LSC'
        ]
        
        if any(kw in first_col for kw in skip_keywords):
            continue
        
        if set(first_col.replace('-', '').replace(' ', '')) == set():
            continue
        
        # Detect course name
        if not re.match(r'^\d', first_col) and len(first_col) > 5 and re.search(r'[a-zA-Z]{3,}', first_col):
            clean_name = first_col.replace('\\&', '&').replace('\\*\\*\\*', '').strip()
            if clean_name and not clean_name.isdigit():
                current_course_name = clean_name
                continue
        
        if not current_course_name:
            continue
        
        try:
            # Find GPA
            gpa = None
            gpa_idx = None
            
            for idx, val in enumerate(parts):
                if val in ['***', '\\*\\*\\*', '.', '']:
                    continue
                try:
                    float_val = float(val)
                    if 0 <= float_val <= 4.5 and float_val != int(float_val):
                        gpa = float_val
                        gpa_idx = idx
                        break
                except ValueError:
                    continue
            
            if gpa is None or gpa_idx is None:
                continue
            
            # Extract grades
            grades = []
            for idx in range(gpa_idx + 1, min(gpa_idx + 8, len(parts))):
                val = parts[idx]
                if val in ['.', '', '***', '\\*\\*\\*']:
                    grades.append(0.0)
                else:
                    try:
                        grades.append(float(val))
                    except ValueError:
                        grades.append(0.0)
            
            while len(grades) < 7:
                grades.append(0.0)
            grades = grades[:7]
            
            if gpa > 0 and sum(grades) > 0:
                chunk = {
                    "course_name": current_course_name,
                    "avg_gpa": round(gpa, 3),
                    "grade_a": round(grades[0], 1),
                    "grade_ab": round(grades[1], 1),
                    "grade_b": round(grades[2], 1),
                    "grade_bc": round(grades[3], 1),
                    "grade_c": round(grades[4], 1),
                    "grade_d": round(grades[5], 1),
                    "grade_f": round(grades[6], 1)
                }
                chunks.append(chunk)
                current_course_name = None
                
        except Exception:
            continue
    
    # Remove duplicates
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk['course_name'], chunk['avg_gpa'])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    
    return unique_chunks


def stage2_markdown_to_chunks(input_file="fullgrades.md", 
                               text_file="chunks.txt",
                               json_file="chunks.json"):
    """Stage 2: Convert Markdown to structured chunks"""
    
    print("\n" + "=" * 80)
    print("STAGE 2: MARKDOWN → CHUNKS")
    print("=" * 80)
    
    input_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"\n📄 Input: {input_file} ({input_size_mb:.2f} MB)")
    
    print("🔄 Parsing markdown...")
    start_time = time.perf_counter()
    
    chunks = parse_markdown_to_chunks(input_file)
    
    # Save text format
    with open(text_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"--- Chunk {i} ---\n")
            f.write(f"Course: {chunk['course_name']}\n\n")
            f.write(f"Average GPA: {chunk['avg_gpa']:.3f}\n\n")
            f.write("Grade Distribution:\n")
            f.write(f"- A: {chunk['grade_a']:.1f}%\n")
            f.write(f"- AB: {chunk['grade_ab']:.1f}%\n")
            f.write(f"- B: {chunk['grade_b']:.1f}%\n")
            f.write(f"- BC: {chunk['grade_bc']:.1f}%\n")
            f.write(f"- C: {chunk['grade_c']:.1f}%\n")
            f.write(f"- D: {chunk['grade_d']:.1f}%\n")
            f.write(f"- F: {chunk['grade_f']:.1f}%\n")
            f.write("="*80 + "\n\n")
    
    # Save JSON format
    json_data = []
    for i, c in enumerate(chunks):
        text_repr = (
            f"Course: {c['course_name']}\n\n"
            f"Average GPA: {c['avg_gpa']:.3f}\n\n"
            f"Grade Distribution:\n"
            f"- A: {c['grade_a']:.1f}%\n"
            f"- AB: {c['grade_ab']:.1f}%\n"
            f"- B: {c['grade_b']:.1f}%\n"
            f"- BC: {c['grade_bc']:.1f}%\n"
            f"- C: {c['grade_c']:.1f}%\n"
            f"- D: {c['grade_d']:.1f}%\n"
            f"- F: {c['grade_f']:.1f}%"
        )
        
        json_data.append({
            "id": f"course_{i}",
            "text": text_repr,
            "metadata": c
        })
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    wall_time = time.perf_counter() - start_time
    
    total_output_mb = (os.path.getsize(text_file) + os.path.getsize(json_file)) / (1024 * 1024)
    avg_chunk_size = os.path.getsize(json_file) / len(chunks) if len(chunks) > 0 else 0
    
    stats = {
        "wall_time_seconds": round(wall_time, 3),
        "num_chunks": len(chunks),
        "output_size_mb": round(total_output_mb, 2),
        "avg_chunk_size_bytes": round(avg_chunk_size, 2),
        "throughput_chunks_per_sec": round(len(chunks) / wall_time, 2)
    }
    
    print(f"✅ Complete! Wall Time: {wall_time:.3f}s | Chunks: {len(chunks)}")
    return stats


# ============================================================================
# STAGE 3: CHUNKS → EMBEDDINGS
# ============================================================================

def stage3_chunks_to_embeddings(chunks_file="chunks.json",
                                 embeddings_txt="embeddings.txt",
                                 embeddings_npy="embeddings.npy"):
    """Stage 3: Generate embeddings from chunks"""
    from sentence_transformers import SentenceTransformer
    
    print("\n" + "=" * 80)
    print("STAGE 3: CHUNKS → EMBEDDINGS")
    print("=" * 80)
    
    print(f"\n📖 Loading chunks from {chunks_file}...")
    with open(chunks_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} chunks")
    
    print("\n📦 Loading embedding model...")
    model_load_start = time.perf_counter()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model_load_time = time.perf_counter() - model_load_start
    print(f"✅ Model loaded in {model_load_time:.3f}s")
    
    sentences = [chunk["text"] for chunk in data]
    
    print("\n🔄 Generating embeddings...")
    inference_start = time.perf_counter()
    embeddings = model.encode(sentences, show_progress_bar=True)
    inference_time = time.perf_counter() - inference_start
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Save embeddings
    with open(embeddings_txt, "w", encoding='utf-8') as f:
        for chunk, emb in zip(data, embeddings):
            f.write(f"ID: {chunk['id']}\n")
            f.write(f"Course: {chunk['metadata']['course_name']}\n")
            f.write(f"Embedding (length {len(emb)}):\n")
            f.write(str(emb) + "\n")
            f.write("=" * 60 + "\n")
    
    np.save(embeddings_npy, embeddings)
    
    total_wall_time = model_load_time + inference_time
    output_size_mb = (os.path.getsize(embeddings_txt) + os.path.getsize(embeddings_npy)) / (1024 * 1024)
    
    stats = {
        "total_wall_time_seconds": round(total_wall_time, 3),
        "model_load_time_seconds": round(model_load_time, 3),
        "inference_time_seconds": round(inference_time, 3),
        "num_embeddings": len(embeddings),
        "embedding_dimension": embeddings.shape[1],
        "output_size_mb": round(output_size_mb, 2),
        "throughput_embeddings_per_sec": round(len(embeddings) / inference_time, 2)
    }
    
    print(f"✅ Complete! Total Time: {total_wall_time:.3f}s")
    print(f"   Model Load: {model_load_time:.3f}s | Inference: {inference_time:.3f}s")
    return stats


# ============================================================================
# STAGE 4: RAG CHATBOT
# ============================================================================

def load_rag_system():
    """Load embeddings, FAISS index, and models for RAG"""
    from sentence_transformers import SentenceTransformer
    import faiss
    from openai import OpenAI
    
    print("Loading RAG system...")
    
    # Load embeddings and chunks
    embeddings = np.load("embeddings.npy").astype('float32')
    with open("chunks.json", 'r') as f:
        chunks = json.load(f)
    
    metadata = [{"id": c['id'], **c['metadata']} for c in chunks]
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    # LLM client
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    print(f"✓ Loaded {len(metadata)} courses")
    print(f"✓ FAISS index ready with {index.ntotal} vectors\n")
    
    return model, index, metadata, client


def retrieve_and_generate(query_text, model, index, metadata, client, k=3):
    """Retrieve courses and generate response with timing"""
    
    timing = {}
    total_start = time.perf_counter()
    
    # Query embedding
    embed_start = time.perf_counter()
    query_emb = model.encode(query_text, convert_to_numpy=True).astype('float32')
    query_vector = query_emb.reshape(1, -1)
    timing['query_embedding_ms'] = (time.perf_counter() - embed_start) * 1000
    
    # FAISS search
    search_start = time.perf_counter()
    distances, indices = index.search(query_vector, k)
    timing['faiss_search_ms'] = (time.perf_counter() - search_start) * 1000
    timing['total_retrieval_ms'] = timing['query_embedding_ms'] + timing['faiss_search_ms']
    
    # Format context
    results = []
    context = "Here are the most relevant courses:\n\n"
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        course = metadata[idx]
        results.append((course, dist))
        context += f"{i}. Course: {course['course_name']}\n"
        context += f"   Average GPA: {course['avg_gpa']:.3f}\n"
        if 'grade_a' in course:
            context += f"   Grade A: {course['grade_a']:.1f}%\n"
        context += f"   Relevance: {1/(1+dist):.3f}\n\n"
    
    # LLM generation
    system_prompt = """You are a course data assistant. Answer questions using only the provided course data. Be concise and accurate."""
    
    user_prompt = f"User Query: {query_text}\n\n{context}\n\nProvide a helpful response."
    
    generation_start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Error: {e}\n\nRetrieved courses:\n" + "\n".join([f"{i+1}. {c[0]['course_name']}" for i, c in enumerate(results)])
    
    timing['llm_generation_ms'] = (time.perf_counter() - generation_start) * 1000
    timing['total_query_latency_ms'] = (time.perf_counter() - total_start) * 1000
    timing['query'] = query_text
    timing['response_length'] = len(answer)
    
    return answer, timing


def stage4_interactive_chatbot():
    """Stage 4: Interactive RAG chatbot with timing"""
    
    print("\n" + "=" * 80)
    print("STAGE 4: INTERACTIVE RAG CHATBOT")
    print("=" * 80)
    print("Ask questions about courses (or 'quit' to exit)\n")
    
    model, index, metadata, client = load_rag_system()
    
    while True:
        try:
            query = input("Your question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
            
            print(f"\n⏱️  Processing query...")
            answer, timing = retrieve_and_generate(query, model, index, metadata, client)
            
            print(f"\n📊 Timing:")
            print(f"   Retrieval:  {timing['total_retrieval_ms']:.2f} ms")
            print(f"   Generation: {timing['llm_generation_ms']:.2f} ms")
            print(f"   Total:      {timing['total_query_latency_ms']:.2f} ms")
            
            print(f"\n📋 RESPONSE:")
            print("-" * 60)
            print(answer)
            print("\n" + "=" * 80 + "\n")
            
            query_timings.append(timing)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Save query stats
    save_query_stats()
    print("Goodbye!")


def save_query_stats():
    """Save query timing statistics"""
    if not query_timings:
        return
    
    retrieval_times = [t['total_retrieval_ms'] for t in query_timings]
    generation_times = [t['llm_generation_ms'] for t in query_timings]
    total_latencies = [t['total_query_latency_ms'] for t in query_timings]
    
    query_stats = {
        "num_queries": len(query_timings),
        "retrieval_ms": {
            "mean": round(np.mean(retrieval_times), 2),
            "median": round(np.median(retrieval_times), 2),
            "p95": round(np.percentile(retrieval_times, 95), 2),
            "p99": round(np.percentile(retrieval_times, 99), 2),
            "min": round(np.min(retrieval_times), 2),
            "max": round(np.max(retrieval_times), 2)
        },
        "generation_ms": {
            "mean": round(np.mean(generation_times), 2),
            "median": round(np.median(generation_times), 2),
            "p95": round(np.percentile(generation_times, 95), 2),
            "p99": round(np.percentile(generation_times, 99), 2),
            "min": round(np.min(generation_times), 2),
            "max": round(np.max(generation_times), 2)
        },
        "total_latency_ms": {
            "mean": round(np.mean(total_latencies), 2),
            "median": round(np.median(total_latencies), 2),
            "p95": round(np.percentile(total_latencies, 95), 2),
            "p99": round(np.percentile(total_latencies, 99), 2),
            "min": round(np.min(total_latencies), 2),
            "max": round(np.max(total_latencies), 2)
        },
        "individual_queries": query_timings
    }
    
    try:
        with open(STATS_FILE, 'r') as f:
            all_stats = json.load(f)
    except FileNotFoundError:
        all_stats = {}
    
    all_stats['stage4_query_processing'] = query_stats
    
    with open(STATS_FILE, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"💾 Query stats saved to {STATS_FILE}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_pipeline():
    """Run stages 1-3 (offline pipeline)"""
    
    print("=" * 80)
    print("RAG PIPELINE - BUILDING INDEX")
    print("=" * 80)
    
    pipeline_start = time.perf_counter()
    
    # Stage 1
    s1_stats = stage1_pdf_to_markdown()
    
    # Stage 2
    s2_stats = stage2_markdown_to_chunks()
    
    # Stage 3
    s3_stats = stage3_chunks_to_embeddings()
    
    total_time = time.perf_counter() - pipeline_start
    
    # Save all stats
    all_stats = {
        "stage1_pdf_to_markdown": s1_stats,
        "stage2_markdown_to_chunks": s2_stats,
        "stage3_chunks_to_embeddings": s3_stats,
        "total_offline_pipeline_seconds": round(total_time, 3)
    }
    
    with open(STATS_FILE, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print("\n" + "=" * 80)
    print("BUILD COMPLETE!")
    print("=" * 80)
    print(f"Total Time: {total_time:.3f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nStats saved to: {STATS_FILE}")
    print(f"\nNext: Run 'python {sys.argv[0]} --chat' to start chatbot")


def show_report():
    """Display performance report"""
    
    try:
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f"❌ No stats found. Run with --build first.")
        return
    
    print("\n" + "=" * 80)
    print("RAG PIPELINE PERFORMANCE REPORT")
    print("=" * 80)
    
    # Stage 1
    if 'stage1_pdf_to_markdown' in stats:
        s1 = stats['stage1_pdf_to_markdown']
        print("\n📄 STAGE 1: PDF → MARKDOWN")
        print("-" * 80)
        print(f"   Wall Time:       {s1['wall_time_seconds']:.3f} seconds")
        print(f"   Input Size:      {s1['input_size_mb']:.2f} MB")
        print(f"   Output Size:     {s1['output_size_mb']:.2f} MB")
        print(f"   Throughput:      {s1['throughput_mb_per_sec']:.3f} MB/s")
    
    # Stage 2
    if 'stage2_markdown_to_chunks' in stats:
        s2 = stats['stage2_markdown_to_chunks']
        print("\n📚 STAGE 2: MARKDOWN → CHUNKS")
        print("-" * 80)
        print(f"   Wall Time:       {s2['wall_time_seconds']:.3f} seconds")
        print(f"   Chunks Created:  {s2['num_chunks']}")
        print(f"   Throughput:      {s2['throughput_chunks_per_sec']:.2f} chunks/sec")
    
    # Stage 3
    if 'stage3_chunks_to_embeddings' in stats:
        s3 = stats['stage3_chunks_to_embeddings']
        print("\n🔢 STAGE 3: CHUNKS → EMBEDDINGS")
        print("-" * 80)
        print(f"   Total Wall Time: {s3['total_wall_time_seconds']:.3f} seconds")
        print(f"     - Model Load:  {s3['model_load_time_seconds']:.3f} seconds")
        print(f"     - Inference:   {s3['inference_time_seconds']:.3f} seconds")
        print(f"   Embeddings:      {s3['num_embeddings']}")
        print(f"   Throughput:      {s3['throughput_embeddings_per_sec']:.2f} embeddings/sec")
    
    # Total offline
    if 'total_offline_pipeline_seconds' in stats:
        total = stats['total_offline_pipeline_seconds']
        print("\n⏱️  TOTAL OFFLINE PIPELINE")
        print("-" * 80)
        print(f"   {total:.3f} seconds ({total/60:.2f} minutes)")
    
    # Stage 4 (if available)
    if 'stage4_query_processing' in stats:
        s4 = stats['stage4_query_processing']
        print("\n🔍 STAGE 4: QUERY PROCESSING")
        print("-" * 80)
        print(f"   Total Queries:   {s4['num_queries']}")
        print(f"\n   Retrieval Latency (ms):")
        print(f"     Mean:   {s4['retrieval_ms']['mean']:.2f}")
        print(f"     P95:    {s4['retrieval_ms']['p95']:.2f}")
        print(f"     P99:    {s4['retrieval_ms']['p99']:.2f}")
        print(f"\n   LLM Generation Latency (ms):")
        print(f"     Mean:   {s4['generation_ms']['mean']:.2f}")
        print(f"     P95:    {s4['generation_ms']['p95']:.2f}")
        print(f"     P99:    {s4['generation_ms']['p99']:.2f}")
        print(f"\n   Total Query Latency (ms):")
        print(f"     Mean:   {s4['total_latency_ms']['mean']:.2f}")
        print(f"     P95:    {s4['total_latency_ms']['p95']:.2f}")
        print(f"     P99:    {s4['total_latency_ms']['p99']:.2f}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  python {sys.argv[0]} --build    # Run offline pipeline (stages 1-3)")
        print(f"  python {sys.argv[0]} --chat     # Run interactive chatbot (stage 4)")
        print(f"  python {sys.argv[0]} --report   # Show performance report")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "--build":
        build_pipeline()
    elif command == "--chat":
        stage4_interactive_chatbot()
    elif command == "--report":
        show_report()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()