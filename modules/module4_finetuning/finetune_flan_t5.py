"""
Module 4B: Fine-tuning FLAN-T5 on MCP Dataset
Fine-tunes FLAN-T5-small to generate syllabus-grounded academic answers.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    T5ForConditionalGeneration, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)
import os

class MCPDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_input_length=512, max_output_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Load JSONL data
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.data.append(sample)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize input (MCP format)
        input_text = sample['input']
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target (expected output)
        target_text = sample['output']
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

def load_datasets(tokenizer):
    """Load train and validation datasets"""
    train_dataset = MCPDataset('data/processed/mcp_train.jsonl', tokenizer)
    val_dataset = MCPDataset('data/processed/mcp_val.jsonl', tokenizer)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Replace padding token id in labels with -100
        labels[labels == 0] = -100
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Replace padding token id in labels with -100
            labels[labels == 0] = -100
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    print("=" * 60)
    print("MODULE 4B: FINE-TUNING FLAN-T5 ON MCP DATASET")
    print("=" * 60)
    
    # Configuration
    model_name = "google/flan-t5-small"
    output_dir = "models/flan_t5_mcp"
    batch_size = 2
    learning_rate = 2e-5
    num_epochs = 6
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("\n1. Loading pretrained FLAN-T5 model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    # Load datasets
    print("\n2. Loading MCP datasets...")
    train_dataset, val_dataset = load_datasets(tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    print(f"\n3. Starting fine-tuning...")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total training steps: {total_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss! Saving model...")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    print(f"\n4. Fine-tuning completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()