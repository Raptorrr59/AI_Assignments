import numpy as np
from gpt_model import GPTModel, GPTConfig
from typing import List, Tuple
import json

class GPTTrainer:
    def __init__(self, model: GPTModel, learning_rate: float = 1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.training_history = []
    
    def compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Ensure targets match logits sequence length
        if targets.shape[1] != seq_len:
            min_len = min(seq_len, targets.shape[1])
            logits = logits[:, :min_len, :]
            targets = targets[:, :min_len]
        
        # Reshape for easier computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Clip target indices to valid range
        targets_flat = np.clip(targets_flat, 0, vocab_size - 1)
        
        # Compute softmax
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Compute cross-entropy loss
        log_probs = np.log(probs + 1e-8)
        loss = -np.mean(log_probs[np.arange(len(targets_flat)), targets_flat])
        
        return loss
    
    def train_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """Single training step with simplified gradient computation"""
        # Forward pass
        logits = self.model.forward(input_ids)
        loss = self.compute_loss(logits, targets)
        
        # Simplified backward pass (gradient approximation)
        # In a real implementation, you'd compute actual gradients
        self.update_parameters_simple()
        
        return loss
    
    def update_parameters_simple(self):
        """Simplified parameter update (placeholder for actual gradient descent)"""
        # This is a simplified update - in practice you'd compute actual gradients
        noise_scale = self.learning_rate * 0.01
        
        # Add small random updates to simulate learning
        for block in self.model.blocks:
            block.attention.W_q += np.random.randn(*block.attention.W_q.shape) * noise_scale
            block.attention.W_k += np.random.randn(*block.attention.W_k.shape) * noise_scale
            block.attention.W_v += np.random.randn(*block.attention.W_v.shape) * noise_scale
            block.attention.W_o += np.random.randn(*block.attention.W_o.shape) * noise_scale
    
    def pretrain(self, dataset: List[np.ndarray], epochs: int = 1, batch_size: int = 4):
        """Pre-training phase"""
        print("Starting Pre-training...")
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                
                for sequence in batch:
                    if len(sequence) > 1:
                        input_ids = sequence[:-1].reshape(1, -1)
                        targets = sequence[1:].reshape(1, -1)
                        
                        loss = self.train_step(input_ids, targets)
                        total_loss += loss
                        num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            self.training_history.append({'phase': 'pretrain', 'epoch': epoch, 'loss': avg_loss})
    
    def supervised_fine_tune(self, sft_dataset: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 1):
        """Supervised Fine-Tuning phase"""
        print("Starting Supervised Fine-Tuning...")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for input_seq, target_seq in sft_dataset:
                input_ids = input_seq.reshape(1, -1)
                targets = target_seq.reshape(1, -1)
                
                loss = self.train_step(input_ids, targets)
                total_loss += loss
            
            avg_loss = total_loss / len(sft_dataset)
            print(f"SFT Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            self.training_history.append({'phase': 'sft', 'epoch': epoch, 'loss': avg_loss})
    
    def reward_model_training(self, rm_dataset: List[Tuple[np.ndarray, float]], epochs: int = 1):
        """Reward Model Training phase"""
        print("Starting Reward Model Training...")
        
        # Simplified reward model training
        for epoch in range(epochs):
            total_reward_loss = 0
            
            for sequence, reward in rm_dataset:
                # Simplified reward prediction
                logits = self.model.forward(sequence.reshape(1, -1))
                predicted_reward = np.mean(logits)  # Simplified reward computation
                reward_loss = (predicted_reward - reward) ** 2
                total_reward_loss += reward_loss
            
            avg_reward_loss = total_reward_loss / len(rm_dataset)
            print(f"RM Epoch {epoch+1}/{epochs}, Average Reward Loss: {avg_reward_loss:.4f}")
            self.training_history.append({'phase': 'rm', 'epoch': epoch, 'loss': avg_reward_loss})
    
    def rlhf_training(self, rlhf_dataset: List[Tuple[np.ndarray, np.ndarray, float]], epochs: int = 1):
        """Reinforcement Learning from Human Feedback phase"""
        print("Starting RLHF Training...")
        
        for epoch in range(epochs):
            total_rlhf_loss = 0
            
            for prompt, response, human_preference in rlhf_dataset:
                # Simplified RLHF training
                input_ids = np.concatenate([prompt, response]).reshape(1, -1)
                logits = self.model.forward(input_ids)
                
                # Simplified policy gradient approximation
                policy_loss = -human_preference * np.mean(logits)
                total_rlhf_loss += abs(policy_loss)
                
                # Simple parameter update
                self.update_parameters_simple()
            
            avg_rlhf_loss = total_rlhf_loss / len(rlhf_dataset)
            print(f"RLHF Epoch {epoch+1}/{epochs}, Average RLHF Loss: {avg_rlhf_loss:.4f}")
            self.training_history.append({'phase': 'rlhf', 'epoch': epoch, 'loss': avg_rlhf_loss})
    
    def save_training_history(self, filepath: str):
        """Save training history"""
        with open(filepath, 'w') as f:
            json.dump(self.training_history, f, indent=2)