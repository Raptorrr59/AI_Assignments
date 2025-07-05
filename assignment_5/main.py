import numpy as np
from gpt_model import GPTModel, GPTConfig
from training import GPTTrainer
from tokenizer import SimpleTokenizer
from evaluation import GPTEvaluator
from typing import List
import json

def create_dummy_dataset(tokenizer: SimpleTokenizer, size: int = 100) -> List[np.ndarray]:
    """Create a dummy dataset for demonstration"""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized the field of NLP."
    ]
    
    dataset = []
    for _ in range(size):
        text = np.random.choice(sample_texts)
        tokens = tokenizer.encode(text)
        if len(tokens) > 5:  # Ensure minimum length
            dataset.append(np.array(tokens[:50]))  # Truncate to max 50 tokens
    
    return dataset

def create_sft_dataset(tokenizer: SimpleTokenizer, size: int = 50):
    """Create supervised fine-tuning dataset"""
    qa_pairs = [
        ("What is AI?", "AI is artificial intelligence, the simulation of human intelligence in machines."),
        ("How does machine learning work?", "Machine learning uses algorithms to learn patterns from data."),
        ("What are neural networks?", "Neural networks are computing systems inspired by biological neural networks.")
    ]
    
    sft_data = []
    for _ in range(size):
        # Fix: Use random index instead of np.random.choice on tuples
        idx = np.random.randint(0, len(qa_pairs))
        q, a = qa_pairs[idx]
        input_tokens = tokenizer.encode(q)
        target_tokens = tokenizer.encode(a)
        
        if len(input_tokens) > 0 and len(target_tokens) > 0:
            sft_data.append((np.array(input_tokens), np.array(target_tokens)))
    
    return sft_data

def create_rm_dataset(tokenizer: SimpleTokenizer, size: int = 30):
    """Create reward model dataset"""
    texts_with_rewards = [
        ("This is a helpful and informative response.", 0.9),
        ("This response is somewhat helpful.", 0.6),
        ("This response is not very helpful.", 0.3),
        ("This is an excellent and detailed explanation.", 0.95),
        ("This response is confusing and unclear.", 0.2)
    ]
    
    rm_data = []
    for _ in range(size):
        # Fix: Use random index instead of np.random.choice on length
        idx = np.random.randint(0, len(texts_with_rewards))
        text, reward = texts_with_rewards[idx]
        tokens = tokenizer.encode(text)
        
        if len(tokens) > 0:
            rm_data.append((np.array(tokens), reward))
    
    return rm_data

def create_rlhf_dataset(tokenizer: SimpleTokenizer, size: int = 20):
    """Create RLHF dataset"""
    rlhf_data = []
    prompts = ["Explain quantum computing", "What is climate change?", "How do computers work?"]
    responses = ["Quantum computing uses quantum mechanics", "Climate change refers to global warming", "Computers process information using circuits"]
    
    for _ in range(size):
        prompt = np.random.choice(prompts)
        response = np.random.choice(responses)
        preference = np.random.uniform(0.3, 1.0)  # Random human preference score
        
        prompt_tokens = tokenizer.encode(prompt)
        response_tokens = tokenizer.encode(response)
        
        if len(prompt_tokens) > 0 and len(response_tokens) > 0:
            rlhf_data.append((np.array(prompt_tokens), np.array(response_tokens), preference))
    
    return rlhf_data

def main():
    print("=== GPT Implementation from Scratch ===")
    print("Initializing model and tokenizer...")
    
    # Initialize configuration
    config = GPTConfig(
        vocab_size=1000,  # Smaller for demo
        max_seq_len=128,
        n_layers=4,       # Smaller for demo
        n_heads=4,
        d_model=256,      # Smaller for demo
        d_ff=1024
    )
    
    # Initialize components
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    model = GPTModel(config)
    trainer = GPTTrainer(model)
    evaluator = GPTEvaluator(model, tokenizer)
    
    print(f"Model initialized with {evaluator.count_parameters():,} parameters")
    
    # Create datasets
    print("\nCreating datasets...")
    pretrain_data = create_dummy_dataset(tokenizer, size=200)
    sft_data = create_sft_dataset(tokenizer, size=50)
    rm_data = create_rm_dataset(tokenizer, size=30)
    rlhf_data = create_rlhf_dataset(tokenizer, size=20)
    test_data = create_dummy_dataset(tokenizer, size=50)
    
    print(f"Created datasets: {len(pretrain_data)} pretrain, {len(sft_data)} SFT, {len(rm_data)} RM, {len(rlhf_data)} RLHF")
    
    # Training pipeline
    print("\n=== Training Pipeline ===")
    
    # 1. Pre-training
    trainer.pretrain(pretrain_data, epochs=2, batch_size=8)
    
    # 2. Supervised Fine-Tuning
    trainer.supervised_fine_tune(sft_data, epochs=1)
    
    # 3. Reward Model Training
    trainer.reward_model_training(rm_data, epochs=1)
    
    # 4. RLHF Training
    trainer.rlhf_training(rlhf_data, epochs=1)
    
    print("\n=== Evaluation ===")
    
    # Evaluation prompts
    eval_prompts = [
        "What is machine learning?",
        "Explain artificial intelligence",
        "How do neural networks work?"
    ]
    
    # Evaluate model
    results = evaluator.evaluate_model(test_data, eval_prompts)
    
    print(f"\nEvaluation Results:")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Model Parameters: {results['model_parameters']:,}")
    print(f"Inference Speed: {results['speed_metrics']['tokens_per_second']:.2f} tokens/sec")
    
    print("\nGenerated Samples:")
    for i, sample in enumerate(results['generated_samples']):
        print(f"Prompt {i+1}: {eval_prompts[i]}")
        print(f"Generated: {sample}")
        print()
    
    # Save everything
    print("Saving model and results...")
    model.save_model('gpt_model.pkl')
    tokenizer.save_vocab('vocab.json')
    trainer.save_training_history('training_history.json')
    evaluator.save_evaluation_results('evaluation_results.json')
    
    print("\n=== Training Complete ===")
    print("Files saved:")
    print("- gpt_model.pkl (model weights)")
    print("- vocab.json (tokenizer vocabulary)")
    print("- training_history.json (training metrics)")
    print("- evaluation_results.json (evaluation metrics)")
    
    return model, trainer, evaluator, results

if __name__ == "__main__":
    model, trainer, evaluator, results = main()