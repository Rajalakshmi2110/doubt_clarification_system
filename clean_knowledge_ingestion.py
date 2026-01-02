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
    source_type: str
    book_priority: int
    source_file: str
    chunk_id: str

class ImprovedKnowledgeIngestion:
    def __init__(self, syllabus_path: str):
        self.syllabus_units = self._load_syllabus_units(syllabus_path)
        self.unit_names = list(self.syllabus_units.keys())
        
        # Patterns to skip non-academic content
        self.skip_patterns = [
            r'table of contents',
            r'acknowledgments?',
            r'about the author',
            r'preface',
            r'copyright',
            r'isbn',
            r'pearson',
            r'references',
            r'index',
            r'bibliography',
            r'appendix',
            r'glossary'
        ]
        
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
        
        units = re.findall(r'UNIT\s+([IVX]+)\s+([^\n]+?)\s*\n([^U]*?)(?=UNIT|$)', text, re.DOTALL)
        unit_map = {}
        
        for roman, title, content in units:
            roman_to_num = {'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5'}
            num = roman_to_num.get(roman, roman)
            
            keywords = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            keywords = list(set([k for k in keywords if k not in ['unit', 'layer', 'protocol', 'network', 'data']]))
            
            unit_map[f"Unit {num}"] = keywords[:10]
        
        return unit_map
    
    def _should_skip_page(self, text: str) -> bool:
        """Check if page should be skipped based on content"""
        text_lower = text.lower()
        
        # Skip if contains skip patterns
        for pattern in self.skip_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Skip if mostly numbers (page numbers, TOC)
        words = text.split()
        if len(words) > 0:
            number_ratio = sum(1 for word in words if word.isdigit()) / len(words)
            if number_ratio > 0.3:
                return True
        
        # Skip very short pages (likely headers/footers)
        if len(text.strip()) < 200:
            return True
            
        return False
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'^[A-Z\s]{10,}\n', '', text, flags=re.MULTILINE)
        
        # Fix broken words (OCR artifacts)
        text = re.sub(r'([a-z])\n([a-z])', r'\1\2', text)
        text = re.sub(r'([A-Z])\n([A-Z])', r'\1\2', text)
        
        # Clean excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_academic_content(self, pdf_path: str) -> str:
        """Extract only academic content from PDF"""
        doc = fitz.open(pdf_path)
        academic_text = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Skip non-academic pages
            if self._should_skip_page(page_text):
                continue
                
            # Clean and add to academic content
            cleaned_text = self._clean_text(page_text)
            if cleaned_text:
                academic_text += cleaned_text + "\n\n"
        
        doc.close()
        return academic_text
    
    def _chunk_by_paragraphs(self, text: str, min_size: int = 200, max_size: int = 800) -> List[str]:
        """Improved semantic chunking"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds max size, save current chunk
            if len(current_chunk) + len(para) > max_size and len(current_chunk) >= min_size:
                chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
            else:
                current_chunk += para + "\n\n"
        
        # Add final chunk if it meets minimum size
        if len(current_chunk.strip()) >= min_size:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _assign_unit(self, chunk_text: str) -> str:
        """Improved unit assignment"""
        chunk_lower = chunk_text.lower()
        best_match = "General"
        max_matches = 0
        
        for unit, keywords in self.syllabus_units.items():
            matches = sum(1 for keyword in keywords if keyword in chunk_lower)
            if matches > max_matches:
                max_matches = matches
                best_match = unit
        
        return best_match
    
    def process_textbook(self, pdf_path: str, book_priority: int) -> List[DocumentChunk]:
        """Process textbook with improved filtering"""
        text = self._extract_academic_content(pdf_path)
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
        """Process notes with improved filtering"""
        text = self._extract_academic_content(pdf_path)
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
        """Save chunks to JSON"""
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
        
        print(f"Saved {len(chunks)} clean chunks to {output_path}")

def main():
    pipeline = ImprovedKnowledgeIngestion("data/raw/syllabus/syllabus.txt")
    
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
            print(f"Processed {len(chunks)} clean chunks from {Path(path).name}")
    
    # Process notes
    notes_dir = Path("data/raw/notes")
    if notes_dir.exists():
        for notes_path in notes_dir.glob("*.pdf"):
            chunks = pipeline.process_notes(str(notes_path))
            all_chunks.extend(chunks)
            print(f"Processed {len(chunks)} clean chunks from {notes_path.name}")
    
    # Save cleaned chunks
    pipeline.save_chunks(all_chunks, "data/processed/knowledge_chunks_clean.json")
    
    # Print summary
    print(f"\nTotal clean chunks: {len(all_chunks)}")
    unit_counts = {}
    for chunk in all_chunks:
        unit_counts[chunk.unit] = unit_counts.get(chunk.unit, 0) + 1
    
    print("\nClean chunks per unit:")
    for unit, count in sorted(unit_counts.items()):
        print(f"  {unit}: {count}")

if __name__ == "__main__":
    main()