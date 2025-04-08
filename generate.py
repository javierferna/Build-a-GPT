import torch
import tiktoken
import argparse
import time
from my_gpt import GPT, Config
import sys

def generate(model, prompt, tokenizer, max_new_tokens=1000, temperature=1.0, top_k=40, use_kv_cache=True):
    """
    Generate text from a prompt using the trained GPT model with KV caching support.
    
    Args:
        model: The trained GPT model
        prompt: The text prompt to start generation
        tokenizer: The tokenizer used to encode/decode text
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        top_k: Number of highest probability tokens to consider for sampling
        use_kv_cache: Whether to use KV caching for more efficient generation
    
    Returns:
        The generated text including the prompt and generation time
    """
    model.eval()  # Set the model to evaluation mode
    
    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt)
    tokens = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(model.lm_head.weight.device)
    
    # Track timing for performance analysis
    start_time = time.time()
    
    # Initialize the past key values to None (no caching yet)
    past_key_values = None
    max_past_key_values_len = model.config.block_size - 1
    
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        
        # For KV cache: after first iteration, we only process the last token
        # For no KV cache: always process full sequence within block size limit
        if not use_kv_cache or past_key_values is None:
            # Get only the last block_size tokens if input is too long
            context = tokens[:, -model.config.block_size:]  
        else:
            context = tokens[:, -1:]  # With KV cache, we only need the last token
            # Get only the last block_size - 1 KV cache if the total input (KV cache + context) is too long
            if past_key_values[0][0].size(2) > max_past_key_values_len:
                past_key_values = list(tuple(t[:, :, -max_past_key_values_len:] for t in layer_past) for layer_past in past_key_values)
                
        # Forward pass to get logits
        with torch.no_grad():
            logits, new_past_key_values = model(context, past_key_values=past_key_values, use_cache=use_kv_cache)
            
            # Update KV cache for next iteration if using cache
            if use_kv_cache:
                past_key_values = new_past_key_values
        
        # Focus on the last token's predictions
        logits = logits[:, -1, :]
        
        # Apply top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            # Set other logits outside top-l to a value of -inf
            logits = torch.where(logits < v[:, -1, None], torch.full_like(logits, float('-inf')), logits)
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
         # If we reach the end of text token, stop
        if next_token.item() == tokenizer.eot_token:
            break
        else:
            # Append the token to our sequence
            tokens = torch.cat((tokens, next_token), dim=1)
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Decode the tokens
    generated_text = tokenizer.decode(tokens[0].tolist())
    
    # Return both the generated text and timing information
    return generated_text, generation_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text with a trained RoPE GPT model')
    parser.add_argument('--prompt', type=str, default="The ship was", help='Text prompt to start generation')
    parser.add_argument('--model_path', type=str, default='models/best_gpt_model.pt', help='Path to the trained model')
    parser.add_argument('--max_tokens', type=int, default=1000, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling parameter')
    parser.add_argument('--use_kv_cache', action='store_true', help='Use KV caching for generation')
    args = parser.parse_args()
    
    # Load the tokenizer (GPT-4 tokenizer)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model configuration (must match the trained model's configuration)
    config = Config(
        vocab_size=tokenizer.n_vocab,
        n_embd=256,
        n_head=8,
        n_layer=4,
        block_size=128
    )
    
    # Initialize the model
    model = GPT(config)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Move model to the appropriate device
    model.to(device)
    
    # Generate text using specified setting
    print(f"\nGenerating text using {'KV cache' if args.use_kv_cache else 'standard generation'}...")
    
    generated_text, generation_time = generate(
        model=model,
        prompt=args.prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_kv_cache=args.use_kv_cache,
    )
    
    # Print timing information
    print(f"Generation completed in {generation_time:.4f} seconds")
    
    # Print the generated text
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
