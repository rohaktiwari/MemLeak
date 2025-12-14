import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset as hf_load_dataset
from transformers import AutoTokenizer
import numpy as np

def get_dataloader(dataset_name, batch_size=64, split='train', max_samples=None, tokenizer_name='distilbert-base-uncased'):
    """
    Unified data loader for Vision (CIFAR10) and Text (AG News, SST2, IMDB).
    """
    
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        if split == 'train':
            ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        else:
            ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            
    elif dataset_name.lower() in ['ag_news', 'sst2', 'imdb']:
        # Load from HuggingFace
        hf_ds = hf_load_dataset(dataset_name, split=split)
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        def tokenize_function(examples):
            # Handle different text column names
            text_col = 'text' if 'text' in examples else 'sentence'
            return tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=128)
            
        ds = hf_ds.map(tokenize_function, batched=True)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
        
    # Subsampling if requested (for quick experiments or shadow models)
    if max_samples and max_samples < len(ds):
        indices = np.random.choice(len(ds), max_samples, replace=False)
        ds = Subset(ds, indices)
        
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'), num_workers=2)

def get_num_classes(dataset_name):
    if dataset_name.lower() == 'cifar10': return 10
    if dataset_name.lower() == 'ag_news': return 4
    if dataset_name.lower() == 'sst2': return 2
    if dataset_name.lower() == 'imdb': return 2
    return 2 # default
