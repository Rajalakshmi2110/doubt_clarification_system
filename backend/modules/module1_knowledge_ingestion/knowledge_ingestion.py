import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    text: str
    unit: str
    source_type: str  # 'textbook' or 'notes'
    book_priority: int  # 1=primary, 2=secondary, 0=notes
    source_file: str
    chunk_id: str

class KnowledgeIngestionPipeline:
    def __init__(self, syllabus_path: str):
        self.syllabus_units = self._load_syllabus_units(syllabus_path)
        self.unit_names = list(self.syllabus_units.keys())
        
    def _load_syllabus_units(self, syllabus_path: str) -> Dict[str, List[str]]:
        """Extract unit names and keywords from syllabus"""
        if syllabus_path.endswith('.txt'):
            with open(syllabus_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            doc = fitz.open(syllabus_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        
        # Extract units with Roman numerals
        units = re.findall(r'UNIT\s+([IVX]+)\s+([^\n]+?)\s*\n([^U]*?)(?=UNIT|$)', text, re.DOTALL)
        unit_map = {}
        
        for roman, title, content in units:
            # Convert Roman to Arabic
            roman_to_num = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5'}
            num = roman_to_num.get(roman, roman)
            
            # Extract keywords from content
            keywords = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            keywords = list(set([k for k in keywords if k not in ['unit', 'layer', 'protocol', 'network', 'data']]))
            
            unit_map[f"Unit {num}"] = keywords[:10]  # Top 10 keywords
        
        return unit_map
    
    def _clean_text(self, text: str) -> str:
        """Remove headers, footers, and noise"""
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        # Remove header/footer patterns
        text = re.sub(r'^[A-Z\s]{10,}\n', '', text, flags=re.MULTILINE)
        return text.strip()
    
    def _filter_non_content_pages(self, text: str) -> str:
        """Remove table of contents, preface, index, and other non-content sections"""
        lines = text.split('\n')
        filtered_lines = []
        skip_section = False
        
        skip_patterns = [
            r'table\s+of\s+contents', r'contents', r'preface', r'about\s+the\s+author',
            r'about\s+author', r'foreword', r'acknowledgment', r'bibliography',
            r'references', r'appendix', r'index', r'glossary', r'copyright', r'isbn'
        ]
        
        import re
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if we should skip this section
            if any(re.search(pattern, line_lower) for pattern in skip_patterns):
                skip_section = True
                continue
            
            # Reset skip if we hit a new chapter/unit
            if re.search(r'(chapter|unit)\s+\d+', line_lower) or len(line_lower) > 100:
                skip_section = False
            
            if not skip_section and len(line.strip()) > 10:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract and clean text from PDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        
        # Apply filtering and cleaning
        text = self._filter_non_content_pages(text)
        return self._clean_text(text)
    
    def _chunk_by_paragraphs(self, text: str, min_chunk_size: int = 200) -> List[str]:
        """Split text into semantic paragraphs with better boundaries"""
        # Split by double newlines first
        sections = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Check if this section starts a new major topic
            is_new_topic = any(pattern in section.lower() for pattern in [
                'unit ', 'chapter ', 'section ', '***', '###', 'definition:', 'example:'
            ])
            
            # If current chunk is getting large or we hit a new topic, finalize it
            if (len(current_chunk) + len(section) > 800) or (is_new_topic and len(current_chunk) > min_chunk_size):
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = section + "\n\n"
            else:
                current_chunk += section + "\n\n"
        
        # Add the last chunk if it meets minimum size
        if len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk.strip())
        elif chunks:  # If too small, append to last chunk
            chunks[-1] += "\n\n" + current_chunk.strip()
        
        return chunks
    
    def _assign_unit(self, chunk_text: str) -> str:
        """Assign unit based on keyword matching and explicit unit mentions"""
        chunk_lower = chunk_text.lower()
        
        # First check for explicit unit mentions
        unit_patterns = {
            "Unit 1": [r'unit\s*i\b', r'unit\s*1\b', r'unit\s*one\b'],
            "Unit 2": [r'unit\s*ii\b', r'unit\s*2\b', r'unit\s*two\b'],
            "Unit 3": [r'unit\s*iii\b', r'unit\s*3\b', r'unit\s*three\b'],
            "Unit 4": [r'unit\s*iv\b', r'unit\s*4\b', r'unit\s*four\b'],
            "Unit 5": [r'unit\s*v\b', r'unit\s*5\b', r'unit\s*five\b']
        }
        
        import re
        for unit, patterns in unit_patterns.items():
            for pattern in patterns:
                if re.search(pattern, chunk_lower):
                    return unit
        
        # Fallback to keyword matching
        best_match = "General"
        max_matches = 0
        
        for unit, keywords in self.syllabus_units.items():
            matches = sum(1 for keyword in keywords if keyword in chunk_lower)
            if matches > max_matches:
                max_matches = matches
                best_match = unit
        
        return best_match
    
    def process_textbook(self, pdf_path: str, book_priority: int) -> List[DocumentChunk]:
        """Process textbook PDF into chunks"""
        text = self._extract_pdf_text(pdf_path)
        chunks = self._chunk_by_paragraphs(text)
        
        document_chunks = []
        filename = Path(pdf_path).stem
        
        for i, chunk in enumerate(chunks):
            unit = self._assign_unit(chunk)
            chunk_obj = DocumentChunk(
                text=chunk,
                unit=unit,
                source_type="textbook",
                book_priority=book_priority,
                source_file=filename,
                chunk_id=f"{filename}_chunk_{i}"
            )
            document_chunks.append(chunk_obj)
        
        return document_chunks
    
    def process_notes(self, pdf_path: str) -> List[DocumentChunk]:
        """Process class notes PDF into chunks"""
        text = self._extract_pdf_text(pdf_path)
        chunks = self._chunk_by_paragraphs(text)
        
        document_chunks = []
        filename = Path(pdf_path).stem
        
        for i, chunk in enumerate(chunks):
            unit = self._assign_unit(chunk)
            chunk_obj = DocumentChunk(
                text=chunk,
                unit=unit,
                source_type="notes",
                book_priority=0,
                source_file=filename,
                chunk_id=f"{filename}_chunk_{i}"
            )
            document_chunks.append(chunk_obj)
        
        return document_chunks
    
    def save_chunks(self, chunks: List[DocumentChunk], output_path: str):
        """Save chunks to JSON file"""
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                'text': chunk.text,
                'unit': chunk.unit,
                'source_type': chunk.source_type,
                'book_priority': chunk.book_priority,
                'source_file': chunk.source_file,
                'chunk_id': chunk.chunk_id
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} chunks to {output_path}")
    
    def save_chunks_by_source(self, chunks: List[DocumentChunk], base_path: str):
        """Save chunks separated by source type and book priority"""
        # Separate chunks by source
        primary_textbook = []
        secondary_textbook = []
        notes = []
        
        for chunk in chunks:
            if chunk.source_type == "textbook" and chunk.book_priority == 1:
                primary_textbook.append(chunk)
            elif chunk.source_type == "textbook" and chunk.book_priority == 2:
                secondary_textbook.append(chunk)
            elif chunk.source_type == "notes":
                notes.append(chunk)
        
        # Save each category separately
        if primary_textbook:
            self._save_chunk_list(primary_textbook, f"{base_path}_primary_textbook.json")
        if secondary_textbook:
            self._save_chunk_list(secondary_textbook, f"{base_path}_secondary_textbook.json")
        if notes:
            self._save_chunk_list(notes, f"{base_path}_notes.json")
    
    def _save_chunk_list(self, chunks: List[DocumentChunk], output_path: str):
        """Helper to save a list of chunks to JSON"""
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                'text': chunk.text,
                'unit': chunk.unit,
                'source_type': chunk.source_type,
                'book_priority': chunk.book_priority,
                'source_file': chunk.source_file,
                'chunk_id': chunk.chunk_id
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} chunks to {output_path}")

def main():
    # Initialize pipeline
    pipeline = KnowledgeIngestionPipeline("data/raw/syllabus/syllabus.txt")
    
    all_chunks = []
    
    # Process textbooks
    textbook_paths = [
        ("data/raw/textbooks/Computer Networking A Top-Down Approach.pdf", 1),
        ("data/raw/textbooks/Data and Computer Communications by William Stallings.pdf", 2)
    ]
    
    for path, priority in textbook_paths:
        if Path(path).exists():
            chunks = pipeline.process_textbook(path, priority)
            all_chunks.extend(chunks)
            print(f"Processed {len(chunks)} chunks from {Path(path).name}")
    
    # Process notes (only PDFs)
    notes_dir = Path("data/raw/notes")
    if notes_dir.exists():
        for notes_path in notes_dir.glob("*.pdf"):
            chunks = pipeline.process_notes(str(notes_path))
            all_chunks.extend(chunks)
            print(f"Processed {len(chunks)} chunks from {notes_path.name}")
    
    # Save all chunks (combined)
    pipeline.save_chunks(all_chunks, "data/processed/knowledge_chunks.json")
    
    # Save chunks separated by source
    pipeline.save_chunks_by_source(all_chunks, "data/processed/knowledge_chunks")
    
    # Print summary
    print(f"\nTotal chunks processed: {len(all_chunks)}")
    unit_counts = {}
    source_counts = {}
    
    for chunk in all_chunks:
        unit_counts[chunk.unit] = unit_counts.get(chunk.unit, 0) + 1
        source_counts[chunk.source_type] = source_counts.get(chunk.source_type, 0) + 1
    
    print("\nChunks per unit:")
    for unit, count in sorted(unit_counts.items()):
        print(f"  {unit}: {count}")
    
    print("\nChunks per source type:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")

if __name__ == "__main__":
    # Install dependencies first
    print("Installing dependencies...")
    import subprocess
    try:
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully!\n")
    except:
        print("Dependency installation failed, continuing anyway...\n")
    
    main()