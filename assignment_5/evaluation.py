import numpy as np
from gpt_model import GPTModel
from tokenizer import SimpleTokenizer
from typing import List, Dict, Tuple
import json
import time

class GPTEvaluator:
    def __init__(self, model: GPTModel, tokenizer: SimpleTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.metrics = {}
    
    def perplexity(self, test_sequences: List[np.ndarray]) -> float:
        """Calculate perplexity on test data"""
        total_log_prob = 0
        total_tokens = 0
        
        for sequence in test_sequences:
            if len(sequence) > 1:
                input_ids = sequence[:-1].reshape(1, -1)
                targets = sequence[1:]
                
                logits = self.model.forward(input_ids)
                
                # Compute log probabilities
                for i, target in enumerate(targets):
                    if i < logits.shape[1]:
                        token_logits = logits[0, i, :]
                        # Softmax
                        exp_logits = np.exp(token_logits - np.max(token_logits))
                        probs = exp_logits / np.sum(exp_logits)
                        
                        if target < len(probs):
                            log_prob = np.log(probs[target] + 1e-8)
                            total_log_prob += log_prob
                            total_tokens += 1
        
        if total_tokens == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_tokens
        perplexity = np.exp(-avg_log_prob)
        return perplexity
    
    def generate_samples(self, prompts: List[str], max_length: int = 50) -> List[str]:
        """Generate text samples for qualitative evaluation"""
        samples = []
        
        for prompt in prompts:
            # Encode prompt
            input_ids = np.array(self.tokenizer.encode(prompt)).reshape(1, -1)
            
            # Generate
            generated_ids = self.model.generate(input_ids, max_length)
            
            # Decode
            generated_text = self.tokenizer.decode(generated_ids[0].tolist())
            samples.append(generated_text)
        
        return samples
    
    def benchmark_speed(self, test_sequences: List[np.ndarray], num_runs: int = 10) -> Dict[str, float]:
        """Benchmark model inference speed"""
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            for sequence in test_sequences[:10]:  # Test on first 10 sequences
                input_ids = sequence[:min(len(sequence), 100)].reshape(1, -1)
                _ = self.model.forward(input_ids)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'tokens_per_second': len(test_sequences[:10]) * 100 / np.mean(times)
        }
    
    def evaluate_model(self, test_data: List[np.ndarray], prompts: List[str]) -> Dict:
        """Comprehensive model evaluation"""
        print("Starting model evaluation...")
        
        # Perplexity
        print("Computing perplexity...")
        ppl = self.perplexity(test_data)
        
        # Text generation samples
        print("Generating text samples...")
        samples = self.generate_samples(prompts)
        
        # Speed benchmark
        print("Benchmarking speed...")
        speed_metrics = self.benchmark_speed(test_data)
        
        # Model size metrics
        total_params = self.count_parameters()
        
        results = {
            'perplexity': ppl,
            'generated_samples': samples,
            'speed_metrics': speed_metrics,
            'model_parameters': total_params,
            'model_config': {
                'vocab_size': self.model.config.vocab_size,
                'max_seq_len': self.model.config.max_seq_len,
                'n_layers': self.model.config.n_layers,
                'n_heads': self.model.config.n_heads,
                'd_model': self.model.config.d_model
            }
        }
        
        self.metrics = results
        return results
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        total = 0
        
        # Embeddings
        total += self.model.token_embedding.size
        total += self.model.position_embedding.size
        total += self.model.lm_head.size
        
        # Transformer blocks
        for block in self.model.blocks:
            total += block.attention.W_q.size
            total += block.attention.W_k.size
            total += block.attention.W_v.size
            total += block.attention.W_o.size
            total += block.feed_forward.W1.size
            total += block.feed_forward.b1.size
            total += block.feed_forward.W2.size
            total += block.feed_forward.b2.size
            total += block.ln1.gamma.size
            total += block.ln1.beta.size
            total += block.ln2.gamma.size
            total += block.ln2.beta.size
        
        return total
    
    def save_evaluation_results(self, filepath: str):
        """Save evaluation results"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def compare_with_baseline(self, baseline_metrics: Dict) -> Dict:
        """Compare with baseline model metrics"""
        comparison = {}
        
        if 'perplexity' in baseline_metrics and 'perplexity' in self.metrics:
            comparison['perplexity_improvement'] = (
                baseline_metrics['perplexity'] - self.metrics['perplexity']
            ) / baseline_metrics['perplexity'] * 100
        
        if 'model_parameters' in baseline_metrics and 'model_parameters' in self.metrics:
            comparison['parameter_ratio'] = self.metrics['model_parameters'] / baseline_metrics['model_parameters']
        
        return comparison