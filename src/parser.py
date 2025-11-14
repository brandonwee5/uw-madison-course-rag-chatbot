import re
import json


def parse_grades(input_file="../data/processed/fullgrades.md"):
    """
    Enhanced parser for UW-Madison grade distribution data.
    Handles complex table formats with multiple sections and naming patterns.
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = []
    lines = content.split('\n')
    
    # State tracking
    current_dept = None
    current_dept_code = None
    current_course_name = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line or '|' not in line:
            i += 1
            continue
        
        # Parse table row
        parts = [p.strip() for p in line.split('|')]
        parts = [p for p in parts if p]  # Remove empty strings
        
        if len(parts) < 3:
            i += 1
            continue
        
        first_col = parts[0]
        
        # Pattern 1: Department header (e.g., "132 AGRONOMY", "130 AGROECOL")
        dept_match = re.match(r'^(\d{3})\s+([A-Z\s&]+)$', first_col)
        if dept_match:
            current_dept_code = dept_match.group(1)
            current_dept = dept_match.group(2).strip()
            i += 1
            continue
        
        # Pattern 2: Section header with no course name (e.g., "ALS", "BSE")
        if len(first_col) <= 5 and first_col.isupper() and not any(char.isdigit() for char in first_col):
            current_dept = first_col
            current_dept_code = None
            i += 1
            continue
        
        # Pattern 3: Skip headers and summary rows
        skip_keywords = [
            'Section', 'Grades', 'Total', 'GPA', 'Ave', 'Avg', '#',
            'Percentage', 'Distribution', 'Office', 'Registrar',
            'University', 'Wisconsin', 'Madison', 'TERM', 'Page',
            'Freshmen', 'Sophomore', 'Junior', 'Senior',
            'Graduate', 'Special', 'Undergraduates', 'Professionals',
            'Summary by Level', 'Dept. Total', 'Course Total',
            'Prof Yr', 'Please note'
        ]
        
        if any(kw in first_col for kw in skip_keywords):
            i += 1
            continue
        
        # Skip separator rows
        if set(first_col.replace('-', '').replace(' ', '')) == set():
            i += 1
            continue
        
        # Pattern 4: Course name only (no numbers, appears before course sections)
        # e.g., "Independent Study", "Cropping Systems"
        if not re.search(r'\d', first_col) and len(first_col) > 5:
            current_course_name = first_col.replace('\\&', '&').strip()
            i += 1
            continue
        
        # Pattern 5: Course section row (contains course code and section)
        # e.g., "299 019", "300 001", "108 A A E"
        
        # Check if this row has grade data
        try:
            # Find numeric columns (enrollment, GPA, grade percentages)
            numeric_data = []
            for idx in range(1, len(parts)):
                val = parts[idx]
                if val in ['***', '\\*\\*\\*', '.', '']:
                    break
                try:
                    numeric_data.append((idx, float(val)))
                except ValueError:
                    pass
            
            if len(numeric_data) < 2:  # Need at least enrollment and GPA
                i += 1
                continue
            
            # Extract enrollment (first integer)
            enrollment = None
            gpa = None
            gpa_idx = None
            
            for idx, val in numeric_data:
                if enrollment is None and val == int(val):
                    enrollment = int(val)
                elif enrollment is not None and gpa is None and 0 <= val <= 4.5:
                    gpa = val
                    gpa_idx = idx
                    break
            
            if gpa is None:
                i += 1
                continue
            
            # Extract course code from first column
            course_code_match = re.search(r'(\d{3}(?:\s+[A-Z0-9\s]+)?)', first_col)
            if course_code_match:
                course_code = course_code_match.group(1).strip()
            else:
                course_code = first_col.strip()
            
            # Build full course identifier
            if current_dept_code:
                full_course_id = f"{current_dept_code} {current_dept} {course_code}"
            elif current_dept:
                full_course_id = f"{current_dept} {course_code}"
            else:
                full_course_id = course_code
            
            # Use stored course name or derive from data
            if current_course_name:
                course_name = current_course_name
            else:
                course_name = course_code
            
            # Extract grade percentages (7 grades after GPA)
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
            
            # Ensure we have exactly 7 grades
            while len(grades) < 7:
                grades.append(0.0)
            grades = grades[:7]
            
            # Only add if we have meaningful data
            if enrollment and enrollment > 0 and gpa > 0:
                chunk = {
                    "course_name": course_name,
                    "course_code": full_course_id,
                    "enrollment": enrollment,
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
            
        except Exception as e:
            # Skip problematic rows
            pass
        
        i += 1
    
    return chunks


def save_chunks(chunks, text_file="chunks.txt", json_file="chunks.json"):
    """Save parsed chunks to text and JSON files."""
    
    # Remove duplicates (same course code and GPA)
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        key = (chunk['course_code'], chunk['avg_gpa'])
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)
    
    chunks = unique_chunks
    
    # Save text format
    with open(text_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"Course: {chunk['course_code']}\n")
            f.write(f"Name: {chunk['course_name']}\n")
            f.write(f"Enrollment: {chunk['enrollment']}\n")
            f.write(f"\nAverage GPA: {chunk['avg_gpa']:.3f}\n\n")
            f.write("Grade Distribution:\n")
            f.write(f"  A:  {chunk['grade_a']:>5.1f}%\n")
            f.write(f"  AB: {chunk['grade_ab']:>5.1f}%\n")
            f.write(f"  B:  {chunk['grade_b']:>5.1f}%\n")
            f.write(f"  BC: {chunk['grade_bc']:>5.1f}%\n")
            f.write(f"  C:  {chunk['grade_c']:>5.1f}%\n")
            f.write(f"  D:  {chunk['grade_d']:>5.1f}%\n")
            f.write(f"  F:  {chunk['grade_f']:>5.1f}%\n")
            f.write("="*80 + "\n\n")
    
    # Save JSON format for RAG
    json_data = []
    for i, c in enumerate(chunks):
        # Create rich text representation for embeddings
        text_repr = (
            f"{c['course_code']}: {c['course_name']}\n\n"
            f"Course Details:\n"
            f"- Average GPA: {c['avg_gpa']:.3f}\n"
            f"- Enrollment: {c['enrollment']} students\n\n"
            f"Grade Distribution:\n"
            f"- A: {c['grade_a']:.1f}% of students\n"
            f"- AB: {c['grade_ab']:.1f}% of students\n"
            f"- B: {c['grade_b']:.1f}% of students\n"
            f"- BC: {c['grade_bc']:.1f}% of students\n"
            f"- C: {c['grade_c']:.1f}% of students\n"
            f"- D: {c['grade_d']:.1f}% of students\n"
            f"- F: {c['grade_f']:.1f}% of students\n\n"
            f"Pass rate: {c['grade_a'] + c['grade_ab'] + c['grade_b'] + c['grade_bc'] + c['grade_c']:.1f}%"
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
    
    # Statistics
    total_enrollment = sum(c['enrollment'] for c in chunks)
    avg_gpa_overall = sum(c['avg_gpa'] * c['enrollment'] for c in chunks) / total_enrollment if total_enrollment > 0 else 0
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total courses: {len(chunks)}")
    print(f"   Total enrollment: {total_enrollment:,}")
    print(f"   Average GPA (weighted): {avg_gpa_overall:.3f}")


def analyze_courses(chunks):
    """Provide analysis of parsed courses."""
    if not chunks:
        return
    
    # Highest and lowest GPA courses
    sorted_by_gpa = sorted(chunks, key=lambda x: x['avg_gpa'], reverse=True)
    
    print("\n🎓 Top 5 Highest GPA Courses:")
    for i, c in enumerate(sorted_by_gpa[:5], 1):
        print(f"   {i}. {c['course_code']}: {c['course_name']}")
        print(f"      GPA: {c['avg_gpa']:.3f} | A: {c['grade_a']:.1f}%")
    
    print("\n📚 Top 5 Lowest GPA Courses:")
    for i, c in enumerate(sorted_by_gpa[-5:], 1):
        print(f"   {i}. {c['course_code']}: {c['course_name']}")
        print(f"      GPA: {c['avg_gpa']:.3f} | A: {c['grade_a']:.1f}%")
    
    # Largest courses
    sorted_by_enrollment = sorted(chunks, key=lambda x: x['enrollment'], reverse=True)
    
    print("\n👥 Top 5 Largest Courses:")
    for i, c in enumerate(sorted_by_enrollment[:5], 1):
        print(f"   {i}. {c['course_code']}: {c['course_name']}")
        print(f"      Enrollment: {c['enrollment']} | GPA: {c['avg_gpa']:.3f}")


if __name__ == "__main__":
    print("=" * 80)
    print("UW-Madison Grade Distribution Parser")
    print("=" * 80)
    
    print("\n📖 Parsing grade distribution data...")
    chunks = parse_grades("data/processed/fullgrades.md")
    
    if chunks:
        save_chunks(chunks)
        analyze_courses(chunks)
        
        # Show sample of parsed courses
        print(f"\n💡 Sample of parsed courses:")
        print("-" * 80)
        for i in range(min(5, len(chunks))):
            c = chunks[i]
            print(f"{i+1}. {c['course_code']}")
            print(f"   Name: {c['course_name']}")
            print(f"   GPA: {c['avg_gpa']:.3f} | Students: {c['enrollment']}")
            print(f"   Grades: A={c['grade_a']:.1f}% AB={c['grade_ab']:.1f}% B={c['grade_b']:.1f}%")
            print()
    else:
        print("❌ No chunks found! Check your input file format.")
        print("\nTroubleshooting:")
        print("1. Make sure 'fullgrades.md' exists in the current directory")
        print("2. Check that the file has the expected table format")
        print("3. Verify the file encoding is UTF-8")