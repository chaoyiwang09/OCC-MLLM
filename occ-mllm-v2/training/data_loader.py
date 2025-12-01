"""
Data Loader for HTFA Training

Handles loading dual-image (original + reconstructed) datasets with text annotations
for training the Hierarchical Trinity Fusion Architecture.

Data Format Expected:
    {
        "image_orig": "path/to/original_image.jpg",
        "image_recon": "path/to/reconstructed_image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nWhat's the object in the hand?"},
            {"from": "gpt", "value": "The object is banana."}
        ],
        "occlusion_mask": "path/to/mask.png" (optional)
    }
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import torchvision.transforms as transforms
from dataclasses import dataclass


class HTFADataset(Dataset):
    """
    Dataset for HTFA training with dual images (original + reconstructed).
    
    Args:
        data_path: Path to JSON file containing data annotations
        image_size: Target image size (default: 448)
        tokenizer: Text tokenizer for processing conversations
        max_seq_length: Maximum sequence length for text (default: 4096)
        use_occlusion_mask: Whether to load occlusion masks (default: False)
        image_transform: Optional custom image transforms
    """
    
    def __init__(
        self,
        data_path: str,
        image_size: int = 448,
        tokenizer = None,
        max_seq_length: int = 4096,
        use_occlusion_mask: bool = False,
        image_transform = None
    ):
        super().__init__()
        
        self.data_path = data_path
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.use_occlusion_mask = use_occlusion_mask
        
        # Load data annotations
        print(f" Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data_list = json.load(f)
        
        print(f" Loaded {len(self.data_list)} samples")
        
        # Image preprocessing
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.image_transform = image_transform
        
        # Mask preprocessing (if used)
        if use_occlusion_mask:
            self.mask_transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
                transforms.ToTensor()
            ])
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image)
            return image_tensor
        except Exception as e:
            print(f"  Error loading image {image_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(3, self.image_size, self.image_size)
    
    def load_mask(self, mask_path: str) -> torch.Tensor:
        """Load and preprocess occlusion mask."""
        try:
            mask = Image.open(mask_path).convert('L')  # Grayscale
            mask_tensor = self.mask_transform(mask)
            return mask_tensor
        except Exception as e:
            print(f"  Error loading mask {mask_path}: {e}")
            # Return ones (no occlusion) as fallback
            return torch.ones(1, self.image_size, self.image_size)
    
    def process_conversation(self, conversations: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Process conversation to extract query and response.
        
        Args:
            conversations: List of conversation turns
        
        Returns:
            query: Human query text
            response: Model response text
        """
        query = ""
        response = ""
        
        for turn in conversations:
            role = turn.get('from', '').lower()
            content = turn.get('value', '')
            
            # Remove <image> placeholder tokens
            content = content.replace('<image>', '').strip()
            
            if role in ['human', 'user']:
                query = content
            elif role in ['gpt', 'assistant']:
                response = content
        
        return query, response
    
    def tokenize_text(self, query: str, response: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize query and response for training.
        
        Args:
            query: Human query text
            response: Model response text
        
        Returns:
            Dict with input_ids, attention_mask, labels
        """
        if self.tokenizer is None:
            # Return dummy tokens if tokenizer not provided
            return {
                'input_ids': torch.zeros(1, dtype=torch.long),
                'attention_mask': torch.ones(1, dtype=torch.long),
                'labels': torch.zeros(1, dtype=torch.long)
            }
        
        # Format: <query> <response>
        full_text = f"{query} {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create labels (same as input_ids, but mask query part)
        # In practice, you'd compute query length and mask it
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dict containing:
                - img_orig: Original image tensor [3, H, W]
                - img_recon: Reconstructed image tensor [3, H, W]
                - occlusion_mask: Occlusion mask tensor [1, H, W] (optional)
                - query: Human query text
                - response: Model response text
                - input_ids: Tokenized input [L]
                - attention_mask: Attention mask [L]
                - labels: Ground truth labels [L]
        """
        item = self.data_list[idx]
        
        # Load images
        img_orig_path = item['image_orig']
        img_recon_path = item['image_recon']
        
        img_orig = self.load_image(img_orig_path)
        img_recon = self.load_image(img_recon_path)
        
        # Load mask if available
        occlusion_mask = None
        if self.use_occlusion_mask and 'occlusion_mask' in item:
            mask_path = item['occlusion_mask']
            occlusion_mask = self.load_mask(mask_path)
        
        # Process conversation
        conversations = item.get('conversations', [])
        query, response = self.process_conversation(conversations)
        
        # Tokenize text
        tokenized = self.tokenize_text(query, response)
        
        # Prepare return dict
        sample = {
            'img_orig': img_orig,
            'img_recon': img_recon,
            'query': query,
            'response': response,
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['labels']
        }
        
        if occlusion_mask is not None:
            sample['occlusion_mask'] = occlusion_mask
        
        return sample


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching HTFA samples.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Batched tensors
    """
    # Stack image tensors
    img_orig = torch.stack([item['img_orig'] for item in batch])
    img_recon = torch.stack([item['img_recon'] for item in batch])
    
    # Stack text tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    batched = {
        'img_orig': img_orig,
        'img_recon': img_recon,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'queries': [item['query'] for item in batch],
        'responses': [item['response'] for item in batch]
    }
    
    # Add occlusion masks if present
    if 'occlusion_mask' in batch[0]:
        occlusion_mask = torch.stack([item['occlusion_mask'] for item in batch])
        batched['occlusion_mask'] = occlusion_mask
    
    return batched


def create_htfa_dataloaders(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    tokenizer = None,
    image_size: int = 448,
    batch_size: int = 2,
    num_workers: int = 2,
    pin_memory: bool = False,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders for HTFA.
    
    Args:
        train_data_path: Path to training data JSON
        val_data_path: Path to validation data JSON (optional)
        tokenizer: Text tokenizer
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        **kwargs: Additional arguments for HTFADataset
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader (None if val_data_path not provided)
    """
    print("=" * 80)
    print("Creating HTFA DataLoaders")
    print("=" * 80)
    
    # Create train dataset
    train_dataset = HTFADataset(
        data_path=train_data_path,
        image_size=image_size,
        tokenizer=tokenizer,
        **kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    print(f" Training loader created: {len(train_dataset)} samples, "
          f"{len(train_loader)} batches")
    
    # Create validation dataset if provided
    val_loader = None
    if val_data_path is not None:
        val_dataset = HTFADataset(
            data_path=val_data_path,
            image_size=image_size,
            tokenizer=tokenizer,
            **kwargs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=False
        )
        
        print(f" Validation loader created: {len(val_dataset)} samples, "
              f"{len(val_loader)} batches")
    
    print("=" * 80)
    
    return train_loader, val_loader


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Testing HTFA Data Loader")
    print("=" * 80)
    
    # Create dummy data for testing
    dummy_data = [
        {
            "image_orig": "dummy_orig_1.jpg",
            "image_recon": "dummy_recon_1.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nWhat object is this?"},
                {"from": "gpt", "value": "This is a banana."}
            ]
        },
        {
            "image_orig": "dummy_orig_2.jpg",
            "image_recon": "dummy_recon_2.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe the object."},
                {"from": "gpt", "value": "A red apple in someone's hand."}
            ]
        }
    ]
    
    # Save dummy data
    dummy_data_path = "test_htfa_data.json"
    with open(dummy_data_path, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2)
    
    print(f" Created dummy data file: {dummy_data_path}")
    
    # Test dataset (without actual images)
    print("\n[Test 1] HTFADataset initialization")
    try:
        dataset = HTFADataset(
            data_path=dummy_data_path,
            image_size=448,
            tokenizer=None,  # No tokenizer for testing
            use_occlusion_mask=False
        )
        print(f" Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"  Dataset test skipped (expected without real images): {e}")
    
    # Test conversation processing
    print("\n[Test 2] Conversation processing")
    conversations = [
        {"from": "human", "value": "<image>\nWhat is this object?"},
        {"from": "gpt", "value": "This is a banana."}
    ]
    
    if 'dataset' in locals():
        query, response = dataset.process_conversation(conversations)
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(" Conversation processing works")
    
    print("\n" + "=" * 80)
    print(" Data loader tests completed!")
    print("=" * 80)
    
    # Clean up
    import os
    if os.path.exists(dummy_data_path):
        os.remove(dummy_data_path)

