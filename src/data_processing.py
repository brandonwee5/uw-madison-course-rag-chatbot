import re
import json


def parse_grades(input_file="output.md"):
    """Simple parser for grade distribution"""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    chunks = []
    
    for line in lines:
        line = line.strip()
        
        if not line or '|' not in line:
            continue
        
        # Parse table rows with data
        parts = [p.strip() for p in line.split('|') if p.strip()]
        
        if len(parts) < 10:
            continue
        
        # First column is the course name or section identifier
        first_col = parts[0]
        
        # Skip header rows and summary rows
        skip_keywords = ['Section', 'Grades', 'Total', 'GPA', 'Ave', '#',
                        'Freshmen', 'Sophomore', 'Junior', 'Senior', 
                        'Graduate', 'Special', 'Undergraduates',
                        'ALS', 'AGRICULTURAL', 'BIOLOGICAL', 'LIFE SCIENCES',
                        'BSE', 'LSC', '108', '112', '120']
        
        if any(kw in first_col for kw in skip_keywords):
            continue
        
        # Skip if first column is just numbers (section numbers without course name)
        if first_col.isdigit() or not re.search(r'[a-zA-Z]', first_col):
            continue
        
        # Check if this looks like a course name (has letters, reasonable length)
        if len(first_col) < 5:
            continue
        
        # Try to extract: enrollment, GPA, then grades
        try:
            # Find enrollment and GPA
            enrollment_idx = None
            gpa_idx = None
            
            for i in range(1, min(5, len(parts))):
                val = parts[i]
                
                if val in ['***', '\\*\\*\\*', '.', '']:
                    break
                
                if enrollment_idx is None and val.isdigit():
                    enrollment_idx = i
                    continue
                
                if enrollment_idx is not None and gpa_idx is None:
                    try:
                        gpa = float(val)
                        if 0 <= gpa <= 4.0:
                            gpa_idx = i
                            break
                    except:
                        pass
            
            if gpa_idx is None:
                continue
            
            gpa = float(parts[gpa_idx])
            
            # Extract grades starting after GPA
            grades_start = gpa_idx + 1
            grades = []
            
            for i in range(grades_start, grades_start + 7):
                if i < len(parts) and parts[i] not in ['.', '', '***', '\\*\\*\\*']:
                    try:
                        grades.append(float(parts[i]))
                    except:
                        grades.append(0.0)
                else:
                    grades.append(0.0)
            
            # Make sure we have 7 grades
            while len(grades) < 7:
                grades.append(0.0)
            
            # Clean up course name
            course_name = first_col.replace('\\&', '&')
            
            chunk = {
                "course_name": course_name,
                "avg_gpa": gpa,
                "grade_a": grades[0],
                "grade_ab": grades[1],
                "grade_b": grades[2],
                "grade_bc": grades[3],
                "grade_c": grades[4],
                "grade_d": grades[5],
                "grade_f": grades[6]
            }
            
            chunks.append(chunk)
            
        except:
            pass
    
    return chunks


def save_chunks(chunks, text_file="chunks.txt", json_file="chunks.json"):
    """Save chunks to files"""
    
    # Save text format
    with open(text_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"Course: {chunk['course_name']}\n\n")
            f.write(f"Average GPA: {chunk['avg_gpa']:.3f}\n\n")
            f.write("Grade Distribution:\n")
            f.write(f"- A: {chunk['grade_a']}%\n")
            f.write(f"- AB: {chunk['grade_ab']}%\n")
            f.write(f"- B: {chunk['grade_b']}%\n")
            f.write(f"- BC: {chunk['grade_bc']}%\n")
            f.write(f"- C: {chunk['grade_c']}%\n")
            f.write(f"- D: {chunk['grade_d']}%\n")
            f.write(f"- F: {chunk['grade_f']}%\n")
            f.write("="*80 + "\n\n")
    
    # Save JSON format
    json_data = [
        {
            "id": f"course_{i}",
            "text": f"Course: {c['course_name']}\n\nAverage GPA: {c['avg_gpa']:.3f}\n\nGrade Distribution:\n- A: {c['grade_a']}%\n- AB: {c['grade_ab']}%\n- B: {c['grade_b']}%\n- BC: {c['grade_bc']}%\n- C: {c['grade_c']}%\n- D: {c['grade_d']}%\n- F: {c['grade_f']}%",
            "metadata": c
        }
        for i, c in enumerate(chunks)
    ]
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Created {len(chunks)} chunks")
    print(f"Saved to {text_file} and {json_file}")


if __name__ == "__main__":
    chunks = parse_grades("output.md")
    if chunks:
        save_chunks(chunks)
        print("\nFirst 3 courses:")
        for i in range(min(3, len(chunks))):
            print(f"{i+1}. {chunks[i]['course_name']} - GPA {chunks[i]['avg_gpa']}")
    else:
        print("No chunks found!")