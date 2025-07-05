import numpy as np
from typing import List, Dict
import json
import re

class SimpleTokenizer:
    """A simple tokenizer for demonstration purposes"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }
        self.build_vocab()
    
    def build_vocab(self):
        """Build a simple vocabulary"""
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
        
        # Add common characters and words
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'
        for i, char in enumerate(chars):
            if len(self.vocab) < self.vocab_size:
                self.vocab[char] = len(self.vocab)
                self.inverse_vocab[len(self.inverse_vocab)] = char
        
        # Add common words
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        for word in common_words:
            if len(self.vocab) < self.vocab_size:
                self.vocab[word] = len(self.vocab)
                self.inverse_vocab[len(self.inverse_vocab)] = word
        
        # Fill remaining slots with dummy tokens
        while len(self.vocab) < self.vocab_size:
            dummy_token = f'<dummy_{len(self.vocab)}>'
            self.vocab[dummy_token] = len(self.vocab)
            self.inverse_vocab[len(self.inverse_vocab)] = dummy_token
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        tokens.append(self.special_tokens['<bos>'])
        
        # Simple word-level tokenization
        words = re.findall(r'\w+|[.,!?;]', text.lower())
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Character-level fallback
                for char in word:
                    if char in self.vocab:
                        tokens.append(self.vocab[char])
                    else:
                        tokens.append(self.special_tokens['<unk>'])
        
        tokens.append(self.special_tokens['<eos>'])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if token not in ['<pad>', '<bos>', '<eos>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def save_vocab(self, filepath: str):
        """Save vocabulary"""
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f, indent=2)
    
    def load_vocab(self, filepath: str):
        """Load vocabulary"""
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}