"""
Module 3: Context-Aware Supervised Training Dataset Generation
Creates MCP-style training data by augmenting Q&A pairs with retrieved textbook context.
"""

import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import os
from typing import List, Dict, Tuple

class MCPDatasetGenerator:
    def __init__(self, config_path: str = "config.json"):
        """Initialize MCP dataset generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load SBERT model (same as Module 2)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Paths
        self.faiss_index_path = "data/processed/textbook_faiss.index"
        self.metadata_path = "data/processed/textbook_metadata.pkl"
        self.qa_dataset_path = "data/dataset/cleaned_questions.json"
        
        # Load FAISS index and metadata
        self.index = faiss.read_index(self.faiss_index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded FAISS index with {self.index.ntotal} textbook chunks")
    
    def retrieve_context(self, question: str, k: int = 4) -> List[Dict]:
        """Retrieve top-k relevant textbook chunks for a question."""
        # Encode question using same model as indexing
        question_embedding = self.model.encode([question])
        
        # Normalize for cosine similarity (same as indexing)
        import faiss
        faiss.normalize_L2(question_embedding.astype('float32'))
        
        # Search FAISS index
        scores, indices = self.index.search(question_embedding.astype('float32'), k)
        
        # Get retrieved chunks with metadata
        retrieved_chunks = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(self.metadata):  # Valid index
                chunk_data = self.metadata[idx]
                # Determine source file based on index position (approximation)
                source_file = "primary_textbook" if idx < 2700 else "secondary_textbook"
                retrieved_chunks.append({
                    'text': chunk_data['text'],
                    'score': float(score),
                    'source_file': source_file,
                    'chunk_id': f"chunk_{idx}",
                    'unit': 'textbook_content'
                })
        
        return retrieved_chunks
    
    def create_mcp_input(self, question: str, context_chunks: List[Dict]) -> str:
        """Create MCP-style input format."""
        # Concatenate retrieved context
        context_text = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # MCP format
        mcp_input = f"""[QUESTION]
{question}

[CONTEXT]
{context_text}

[INSTRUCTION]
Answer the question academically using only the given context."""
        
        return mcp_input
    
    def generate_mcp_dataset(self) -> List[Dict]:
        """Generate complete MCP dataset from Q&A pairs."""
        # Load Q&A dataset
        with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        mcp_dataset = []
        
        for qa_pair in qa_data:
            question = qa_pair['question']
            answer = qa_pair['answer']
            
            # Retrieve relevant context
            context_chunks = self.retrieve_context(question, k=4)
            
            # Create MCP input
            mcp_input = self.create_mcp_input(question, context_chunks)
            
            # Create training example
            training_example = {
                'input': mcp_input,
                'output': answer,
                'metadata': {
                    'question_id': qa_pair['id'],
                    'topic': qa_pair['topic'],
                    'difficulty': qa_pair['difficulty'],
                    'retrieved_chunks': [
                        {
                            'chunk_id': chunk['chunk_id'],
                            'source_file': chunk['source_file'],
                            'unit': chunk['unit'],
                            'score': chunk['score']
                        } for chunk in context_chunks
                    ],
                    'num_chunks_retrieved': len(context_chunks)
                }
            }
            
            mcp_dataset.append(training_example)
            
            if len(mcp_dataset) % 10 == 0:
                print(f"Processed {len(mcp_dataset)} Q&A pairs...")
        
        print(f"Generated {len(mcp_dataset)} MCP training examples")
        return mcp_dataset
    
    def split_and_save_dataset(self, dataset: List[Dict], output_dir: str = "data/processed"):
        """Split dataset and save in JSONL format."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Split: 80% train, 10% val, 10% test
        train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # Save splits
        splits = {
            'train': (train_data, f"{output_dir}/mcp_train.jsonl"),
            'val': (val_data, f"{output_dir}/mcp_val.jsonl"),
            'test': (test_data, f"{output_dir}/mcp_test.jsonl")
        }
        
        for split_name, (data, filepath) in splits.items():
            with open(filepath, 'w', encoding='utf-8') as f:
                for example in data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            print(f"Saved {len(data)} examples to {filepath}")
        
        # Save summary
        summary = {
            'total_examples': len(dataset),
            'train_examples': len(train_data),
            'val_examples': len(val_data),
            'test_examples': len(test_data),
            'split_ratio': '80:10:10',
            'retrieval_k': 4,
            'model_used': 'all-MiniLM-L6-v2'
        }
        
        with open(f"{output_dir}/mcp_dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main execution function."""
    print("=== Module 3: MCP Dataset Generation ===")
    
    # Initialize generator
    generator = MCPDatasetGenerator()
    
    # Generate MCP dataset
    mcp_dataset = generator.generate_mcp_dataset()
    
    # Split and save
    summary = generator.split_and_save_dataset(mcp_dataset)
    
    print("\n=== Dataset Generation Complete ===")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Train: {summary['train_examples']}")
    print(f"Validation: {summary['val_examples']}")
    print(f"Test: {summary['test_examples']}")

if __name__ == "__main__":
    main()