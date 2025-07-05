import numpy as np
import json
import pickle
from typing import List, Tuple, Dict, Optional
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
        # Simplified backward pass
        return grad_output

class MultiHeadAttention:
    def __init__(self, config: GPTConfig):
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.n_heads
        
        # Weight matrices
        self.W_q = np.random.randn(config.d_model, config.d_model) * 0.02
        self.W_k = np.random.randn(config.d_model, config.d_model) * 0.02
        self.W_v = np.random.randn(config.d_model, config.d_model) * 0.02
        self.W_o = np.random.randn(config.d_model, config.d_model) * 0.02
        
        # Create causal mask
        self.register_causal_mask(config.max_seq_len)
    
    def register_causal_mask(self, seq_len: int):
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        self.causal_mask = mask == 1
    
    def scaled_dot_product_attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        # Alternative fix: Use full transpose specification
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        
        attention_weights = self.softmax(scores)
        output = np.matmul(attention_weights, V)
        return output, attention_weights
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Apply attention
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Final linear projection
        output = np.matmul(attn_output, self.W_o)
        return output

class FeedForward:
    def __init__(self, config: GPTConfig):
        self.W1 = np.random.randn(config.d_model, config.d_ff) * 0.02
        self.b1 = np.zeros(config.d_ff)
        self.W2 = np.random.randn(config.d_ff, config.d_model) * 0.02
        self.b2 = np.zeros(config.d_model)
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = self.gelu(np.matmul(x, self.W1) + self.b1)
        output = np.matmul(hidden, self.W2) + self.b2
        return output

class TransformerBlock:
    def __init__(self, config: GPTConfig):
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = LayerNorm(config.d_model, config.layer_norm_eps)
        self.ln2 = LayerNorm(config.d_model, config.layer_norm_eps)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Pre-norm architecture
        attn_output = self.attention.forward(self.ln1.forward(x))
        x = x + attn_output  # Residual connection
        
        ff_output = self.feed_forward.forward(self.ln2.forward(x))
        x = x + ff_output  # Residual connection
        
        return x

class GPTModel:
    def __init__(self, config: GPTConfig):
        self.config = config
        
        # Embedding layers
        self.token_embedding = np.random.randn(config.vocab_size, config.d_model) * 0.02
        self.position_embedding = np.random.randn(config.max_seq_len, config.d_model) * 0.02
        
        # Transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]
        
        # Final layer norm and output projection
        self.ln_f = LayerNorm(config.d_model, config.layer_norm_eps)
        self.lm_head = np.random.randn(config.d_model, config.vocab_size) * 0.02
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding[input_ids]
        
        # Position embeddings
        positions = np.arange(seq_len)
        pos_embeds = self.position_embedding[positions]
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Final layer norm
        x = self.ln_f.forward(x)
        
        # Language modeling head
        logits = np.matmul(x, self.lm_head)
        
        return logits
    
    def generate(self, input_ids: np.ndarray, max_length: int = 50, temperature: float = 1.0) -> np.ndarray:
        """Generate text using the model"""
        generated = input_ids.copy()
        
        for _ in range(max_length):
            # Get logits for the last token
            logits = self.forward(generated)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = np.exp(next_token_logits - np.max(next_token_logits))
            probs = probs / np.sum(probs)
            
            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)
            
            # Add to sequence
            generated = np.concatenate([generated, [[next_token]]], axis=1)
            
            # Stop if we hit max sequence length
            if generated.shape[1] >= self.config.max_seq_len:
                break
        
        return generated
    
    def save_model(self, filepath: str):
        """Save model parameters"""
        model_data = {
            'config': self.config,
            'token_embedding': self.token_embedding,
            'position_embedding': self.position_embedding,
            'lm_head': self.lm_head
        }
        
        # Save transformer blocks
        for i, block in enumerate(self.blocks):
            model_data[f'block_{i}_attention_W_q'] = block.attention.W_q
            model_data[f'block_{i}_attention_W_k'] = block.attention.W_k
            model_data[f'block_{i}_attention_W_v'] = block.attention.W_v
            model_data[f'block_{i}_attention_W_o'] = block.attention.W_o
            model_data[f'block_{i}_ff_W1'] = block.feed_forward.W1
            model_data[f'block_{i}_ff_b1'] = block.feed_forward.b1
            model_data[f'block_{i}_ff_W2'] = block.feed_forward.W2
            model_data[f'block_{i}_ff_b2'] = block.feed_forward.b2
            model_data[f'block_{i}_ln1_gamma'] = block.ln1.gamma
            model_data[f'block_{i}_ln1_beta'] = block.ln1.beta
            model_data[f'block_{i}_ln2_gamma'] = block.ln2.gamma
            model_data[f'block_{i}_ln2_beta'] = block.ln2.beta
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model parameters"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.token_embedding = model_data['token_embedding']
        self.position_embedding = model_data['position_embedding']
        self.lm_head = model_data['lm_head']
        
        # Load transformer blocks
        for i, block in enumerate(self.blocks):
            block.attention.W_q = model_data[f'block_{i}_attention_W_q']
            block.attention.W_k = model_data[f'block_{i}_attention_W_k']
            block.attention.W_v = model_data[f'block_{i}_attention_W_v']
            block.attention.W_o = model_data[f'block_{i}_attention_W_o']
            block.feed_forward.W1 = model_data[f'block_{i}_ff_W1']
            block.feed_forward.b1 = model_data[f'block_{i}_ff_b1']
            block.feed_forward.W2 = model_data[f'block_{i}_ff_W2']
            block.feed_forward.b2 = model_data[f'block_{i}_ff_b2']
            block.ln1.gamma = model_data[f'block_{i}_ln1_gamma']
            block.ln1.beta = model_data[f'block_{i}_ln1_beta']
            block.ln2.gamma = model_data[f'block_{i}_ln2_gamma']
            block.ln2.beta = model_data[f'block_{i}_ln2_beta']