import re
import json


def parse_grades(input_file="../data/processed/fullgrades.md"):
    """
    Parser for UW-Madison grade distribution data.
    Extracts course names, GPA, and grade distributions.
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = []
    lines = content.split('\n')
    
    # Track current course name
    current_course_name = None
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and non-table rows
        if not line or '|' not in line:
            continue
        
        # Parse table row
        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]  # Remove empty strings
        
        if len(parts) < 5:
            continue
        
        first_col = parts[0]
        
        # Skip headers and summary rows
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
        
        # Skip separator rows
        if set(first_col.replace('-', '').replace(' ', '')) == set():
            continue
        
        # Pattern: Course name (no numbers at start, longer than 5 chars)
        # Examples: "Biology of Microorganisms", "Intro Disease Biology"
        if not re.match(r'^\d', first_col) and len(first_col) > 5 and re.search(r'[a-zA-Z]{3,}', first_col):
            # Clean up course name
            clean_name = first_col.replace('\\&', '&').replace('\\*\\*\\*', '').strip()
            if clean_name and not clean_name.isdigit():
                current_course_name = clean_name
                continue
        
        # Try to extract GPA and grades from this row
        # Skip if no course name has been set
        if not current_course_name:
            continue
        
        try:
            # Find GPA in the row (should be between 0 and 4.5)
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
            
            # Extract grade percentages after GPA
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
            
            # Need at least 7 grade values
            while len(grades) < 7:
                grades.append(0.0)
            grades = grades[:7]
            
            # Only add if this looks valid (GPA > 0 and some grades present)
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
                # Reset course name after extracting data
                current_course_name = None
                
        except Exception:
            continue
    
    return chunks


def save_chunks(chunks, text_file="chunks.txt", json_file="chunks.json"):
    """Save parsed chunks to text and JSON files."""
    
    # Remove duplicates based on course name and GPA
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk['course_name'], chunk['avg_gpa'])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    
    chunks = unique_chunks
    
    # Save text format (exactly as requested)
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
    
    # Save JSON format for RAG
    json_data = []
    for i, c in enumerate(chunks):
        # Simple text representation for embeddings
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
    
    print(f"\n✅ Created {len(chunks)} unique course chunks")
    print(f"✅ Saved to {text_file} and {json_file}")


def show_samples(chunks, n=5):
    """Display sample parsed courses."""
    print(f"\n📚 Sample of {min(n, len(chunks))} parsed courses:")
    print("-" * 80)
    
    for i in range(min(n, len(chunks))):
        c = chunks[i]
        print(f"{i+1}. {c['course_name']}")
        print(f"   GPA: {c['avg_gpa']:.3f}")
        print(f"   Grade A: {c['grade_a']:.1f}%")
        print()


def analyze_gpas(chunks):
    """Show GPA statistics."""
    if not chunks:
        return
    
    gpas = [c['avg_gpa'] for c in chunks]
    avg_gpa = sum(gpas) / len(gpas)
    
    print(f"\n📊 GPA Statistics:")
    print(f"   Total courses: {len(chunks)}")
    print(f"   Average GPA: {avg_gpa:.3f}")
    print(f"   Highest GPA: {max(gpas):.3f}")
    print(f"   Lowest GPA: {min(gpas):.3f}")
    
    # Show highest and lowest GPA courses
    sorted_chunks = sorted(chunks, key=lambda x: x['avg_gpa'], reverse=True)
    
    print(f"\n🏆 Top 3 Highest GPA Courses:")
    for i, c in enumerate(sorted_chunks[:3], 1):
        print(f"   {i}. {c['course_name']}: {c['avg_gpa']:.3f}")
    
    print(f"\n📉 Top 3 Lowest GPA Courses:")
    for i, c in enumerate(sorted_chunks[-3:], 1):
        print(f"   {i}. {c['course_name']}: {c['avg_gpa']:.3f}")


if __name__ == "__main__":
    print("=" * 80)
    print("UW-Madison Grade Distribution Parser (Simple)")
    print("=" * 80)
    
    print("\n📖 Parsing grade distribution data...")
    chunks = parse_grades("data/processed/fullgrades.md")
    
    if chunks:
        save_chunks(chunks)
        show_samples(chunks)
        analyze_gpas(chunks)
    else:
        print("\n❌ No chunks found!")
        print("\nTroubleshooting:")
        print("1. Make sure 'fullgrades.md' exists")
        print("2. Check that the file has course names followed by grade data")
        print("3. Verify the file uses pipe-delimited tables")