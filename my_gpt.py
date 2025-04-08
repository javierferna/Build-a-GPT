import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken
import argparse
import os
import re

class Config:
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=4, block_size=128, resid_pdrop=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.resid_pdrop = resid_pdrop


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()


        self.tokenizer = tokenizer
        # Context size for the model
        self.block_size = block_size

        # Process data to add EOT tokens at paragraph endings
        processed_data = self.add_eot_to_paragraphs(self.data)
        
        # Tokenize the processed data - allowing the EOT token
        self.tokens = self.tokenizer.encode(processed_data, allowed_special={"<|endoftext|>"})

    def add_eot_to_paragraphs(self, text):
        """
        Adds an End-Of-Token (EOT) token at the end of each paragraph.
        A paragraph is defined as a block of text separated by double newlines.
        This is needed to ensure that the model will learn to stop generating text after the paragraph is over.
        
        Args:
            text (str): The input text.

        Returns:
            str: The processed text with EOT tokens added at paragraph endings.
        
        """
        # Define the EOT token as a string
        eot_token = "<|endoftext|>"
        
        # Split the text by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Add EOT token at the end of each paragraph
        processed_paragraphs = [p + eot_token for p in paragraphs]
        
        # Join the paragraphs back together
        return '\n\n'.join(processed_paragraphs)


    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        """
        Tokenize the input and output sequences
        
        Args:
            idx (int): The index of the sequence to be returned.

        Returns:
            x (torch.Tensor): The input sequence of shape (block_size,) with dtype torch.long.
            y (torch.Tensor): The output sequence of shape (block_size,) with dtype torch.long, which is a 1-token shifted version of x.
        """
        x, y = torch.tensor(self.tokens[idx : idx + self.block_size], dtype=torch.long), torch.tensor(self.tokens[idx + 1 : idx + self.block_size + 1], dtype=torch.long)

        return x, y


# Rotary Positional Embedding implementation
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        # Initialize fixed rotation frequencies
        inv_freq = 1.0 / (1000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, seq_len):
        """
        This forward pass generates the sine and cosine components of the rotary positional embedding.
        
        Args:
            seq_len: Length of the sequence for which to generate positional encoding.
        
        Returns:
            sin: Sine component of the positional encoding.
            cos: Cosine component of the positional encoding.
        
        """
        position = torch.arange(seq_len, device=self.inv_freq.device)
        
        # We compute the t in sin(t) and cos(t), using the position ids (position) and rotation frequencies (inv_freq)
        # Shape: (seq_len, dim/2)
        position = position.unsqueeze(1)
        t = position * self.inv_freq 
        
        # Compute sin and cos for positional encoding
        # Shape: (seq_len, dim/2)
        sin = torch.sin(t)
        cos = torch.cos(t)

        # Return sin and cos tensors
        return sin, cos


# Helper function to apply rotary embeddings
def apply_rotary_emb(q, k, sin, cos):
    """
    Applies rotary embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        sin: Sine component of the positional encoding
        cos: Cosine component of the positional encoding
    
    Returns:
        q_rotated: Rotated query tensor
        k_rotated: Rotated key tensor
    
    """

    # Split q and k into even and odd indices
    q_even, q_odd = q[..., ::2], q[..., 1::2]  
    k_even, k_odd = k[..., ::2], k[..., 1::2]

    # Apply rotation using the rotation matrix:
    # [cos, -sin]
    # [sin, cos]

    # Rotate query states (q) and key states (k) by sin and cos.
    # If q's seq_len == 1, this means we are using KV cache. 
    # Apply the last position's sin and cos to q
    if q.size(2) != 1:
        # Rotate q normally in training, and in the prefill stage (i.e. first forward pass) during generation.
        # Here q has same seq_len as k.
        q_rotated = torch.cat((q_even * cos - q_odd * sin, q_odd * cos + q_even * sin), dim=-1)
    else:
        # Here is when we are generating new tokens via KV cache. Now q is only one token - the latest token
        # You need to properly slice sin and cos and then rotate q. Make sure you use the current position index.
        q_rotated = torch.cat([
                q_even * cos[:, :, -1:, :] - q_odd * sin[:, :, -1:, :],
                q_odd * cos[:, :, -1:, :] + q_even * sin[:, :, -1:, :]
            ], dim=-1)
    
    # Rotate key states (k). Note that k will always has the full sequence length (i.e. during generation, KV cache +
    #     1 new token), so no need for special handling
    k_rotated = torch.cat((k_even * cos - k_odd * sin, k_odd * cos + k_even * sin), dim=-1)
    
    return q_rotated, k_rotated


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # The W_Q, W_K, W_V matrices into one big matrix
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.attn_dropout = nn.Dropout(config.resid_pdrop)
        # This is the W_Z (or sometimes also called W_O) matrix, which projects the concatenated 
        #     Attention(QK^T)V back to the original embedding dimension
        self.W_O = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Number of attention heads
        self.n_head = config.n_head
        # Split the embedding dimension into heads and split size
        self.split_size = config.n_embd // config.n_head
        # Context length
        self.block_size = config.block_size
        
        # Initialize rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.split_size)
        
        # Register a causal mask buffer that directly masks the attention scores
        # This is more efficient as we apply it once to the scores
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        mask = mask.view(1, 1, config.block_size, config.block_size)
        
        # Add to buffer: Buffers move with the model to CPU and GPU but are not part of the model's state_dict
        # They are also not updated during training.
        self.register_buffer("mask", mask)

    def forward(self, x, layer_past=None, use_cache=False):
        """
        Forward pass for Causal Self Attention with KV cache support.
        
        Args:
            x: input tensor of shape (B, T, C), where B is batch size, T is sequence length, and C is embedding dimension.
            layer_past: Optional tuple (k_cache, v_cache) from previous forward pass for KV caching.
            use_cache: Whether to use and return KV cache.
        
        Returns:
            y: output tensor of shape (B, T, C)
            present: tuple (k_cache, v_cache) to be used in future forward passes if use_cache=True, otherwise None.
        
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        
        # Calculate query, key, values for all heads in batch via W_Q, W_K, W_V projection. (c_attn)
        qkv = self.c_attn(x) 
        q, k, v = qkv.split(C, dim=2) 

        # Reshape q, k, v to (B, nh, T, hs)
        nh = self.n_head
        hs = self.split_size
        q = q.view(B, T, nh, hs).transpose(1, 2)  
        k = k.view(B, T, nh, hs).transpose(1, 2)  
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # Apply KV caching logic if needed
        if layer_past is not None:
            past_k, past_v = layer_past
            # Concatenate past and current key, value tensors
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        
        # Store current keys and values for future use if caching
        present = (k, v) if use_cache else None
        
        # Get current sequence length after potentially concatenating with past
        seq_len = k.size(2)  # Could be T or larger if using KV cache

        # Apply rotary positional embeddings
        sin, cos = self.rotary_emb(seq_len)  

        # Expand sin/cos to match batch and head dimensions - (1, 1, seq_len, hs). Then apply sin/cos to q and k.
        sin = sin.unsqueeze(0).unsqueeze(0)  
        cos = cos.unsqueeze(0).unsqueeze(0)

        q, k = apply_rotary_emb(q, k, sin, cos)  

        # Compute attention scores
        # (B, nh, T, hs) x (B, nh, hs, seq_len) -> (B, nh, T, seq_len)
        
        attn_scores = (q @ k.transpose(-2, -1)) / (hs ** 0.5)
        
        # Apply causal mask - using the appropriate size for this sequence
        # When using KV cache during generation, no need to mask because 
        # The new token's query should be allowed to attend to all past and current keys 
        if layer_past is None:
            mask = self.mask[:, :, :T, :T] 
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention weights to values
        y = attn_weights @ v  
        
        # Re-assemble all head outputs side by side to obtain shape (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        
        # Output projection with W_O matrix, and then pass through residual dropout.
        y = self.resid_dropout(self.W_O(y))
        
        return y, present


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            # Layers for MLP
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, use_cache=False):
        """
        Forward pass with KV cache support.
        
        Args:
            x: input tensor of shape (B, T, C)
            layer_past: Optional KV cache from previous calls
            use_cache: Whether to use and return cache
        
        Returns:
            x: Output tensor
            present: KV cache if use_cache=True, else None
        """
        # Pre-norm and attention with KV cache support
        attn_output, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + attn_output
        
        # Pre-norm and MLP
        x = x + self.mlp(self.ln_2(x))
        
        return x, present


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # We now use ModuleDict to store the transformer components
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        # Language model head - maps the hidden state to logits over vocabulary tokens
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, past_key_values=None, use_cache=False):
        """
        Forward pass for GPT model with KV cache support.
        
        Args: 
            idx (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing token indices.
            past_key_values (list): Optional list of cached key, value tensors from previous forward passes.
            use_cache (bool): Whether to use KV caching and return KV cache.
        
        Returns:
            logits (torch.Tensor): A tensor of shape (batch_size, sequence_length, vocab_size).
            present_key_values (list): List of KV caches for each layer if use_cache=True, else None.
        """
        # Check dimensions
        b, t = idx.size()
        
        # Initialize or use provided cache
        if past_key_values is None:
            past_key_values = [None] * len(self.transformer.h)
            # For full sequence without cache, enforce block size limit
            assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        else:
            # When using KV cache, we only need to check if the total sequence length exceeds block size
            assert len(past_key_values) == len(self.transformer.h), "Past key values length must match number of layers"
            if past_key_values[0] is not None:
                # Get the sequence length from the first layer's KV cache
                past_length = past_key_values[0][0].size(2)
                assert past_length + t <= self.config.block_size, f"Cannot forward sequence of total length {past_length + t}, block size is only {self.config.block_size}"

        # Embedding lookup
        h = self.transformer.wte(idx)
        
        # Store new key-value pairs for each layer
        present_key_values = [] if use_cache else None
        
        # Apply transformer blocks with KV cache
        for i, block in enumerate(self.transformer.h):
            layer_past = past_key_values[i] if past_key_values[i] is not None else None
            h, present = block(h, layer_past=layer_past, use_cache=use_cache)
            if use_cache:
                present_key_values.append(present)
        
        # Apply final layer norm and linear projection to logits
        h = self.transformer.ln_f(h)
        logits = self.lm_head(h)

        return logits, present_key_values


# Function to evaluate the model on validation data
def evaluate(model, validation_loader, criterion, device):
    """
    Evaluate the model on validation data.
    
    Args:
        model: The GPT model to evaluate.
        validation_loader: DataLoader for the validation dataset.
        criterion: Loss function to use for evaluation.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').
    
    Returns:
        avg_loss: Average loss over the validation dataset.
    
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * x.size(0)
    
    avg_loss = total_loss / (len(validation_loader.dataset) * validation_loader.batch_size)
    return avg_loss


# Load the tokenizer (GPT-4 tokenizer)
# This tokenizer has 128k tokens in the vocabulary.
tokenizer = tiktoken.get_encoding("cl100k_base")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a RoPE GPT model')
    parser.add_argument('--file_path', type=str, default='input.txt', help='Path to the input text file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--block_size', type=int, default=128, help='Block size for training')
    parser.add_argument('--train_split', type=float, default=0.9, help='Fraction of data to use for training')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of epochs to train for')

    args = parser.parse_args()

    file_path = args.file_path
    
    # Hyperparameters
    batch_size = args.batch_size
    block_size = args.block_size
    train_split = args.train_split  # E.g. 90% for training, 10% for validationing
    num_epochs = args.num_epochs
    
    # Create the full dataset
    full_dataset = TextDataset(file_path, tokenizer, block_size=block_size)
    
    # Calculate the split sizes
    train_size = int(len(full_dataset) * train_split)
    validation_size = len(full_dataset) - train_size
    
    # Split the dataset
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_size])
    
    # Create data loaders, drop the last batch in training if it's not full.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset split: {train_size} training samples, {validation_size} validationing samples")
    
    # Initialize the model
    config = Config(vocab_size=tokenizer.n_vocab, n_embd=256, n_head=8, n_layer=4, block_size=block_size)
    
    model = GPT(config)
    # Print the number of parameters in the model
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Size of the model {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3):.2f}GB")
    
    # Move the model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device, non_blocking=True)

    # Negative log-likelihood loss function and AdamW optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    
    # Initialize best validation loss tracker and model path
    best_val_loss = float('inf')
    os.mkdir('models') if not os.path.exists('models') else None
    best_model_path = 'models/best_gpt_model.pt'
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for step, (x, y) in enumerate(train_loader):
            # Load data to GPU and clear gradients
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # More memory efficient

            # Forward pass
            logits, _ = model(x)
            # Loss (negative log-likelihood of the predicted token given the true token)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            # Backward pass to compute gradients
            loss.backward()

            # Update parameters using the computed gradients
            optimizer.step()
            # Accumulate training loss
            total_train_loss += loss.item()
            
            # Print training loss every 100 steps
            if step % 100 == 0:
                # Get GPU memory usage
                current_memory = torch.cuda.memory_allocated()
                max_memory = torch.cuda.max_memory_allocated()
                print(f"Current/Maximum GPU memory used: {current_memory/(1024**3):.2f}/{max_memory/(1024**3):.2f} GB")
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Evaluate on the validation set after each epoch
        validation_loss = evaluate(model, validation_loader, criterion, device)
        
        # Print epoch statistics
        avg_train_loss = total_train_loss / (len(train_loader)*train_loader.batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}] complete - Train Loss (per sample): {avg_train_loss:.4f}, Validation Loss (per sample): {validation_loss:.4f}")
        
        # Save the model if it has the best validation loss so far
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    print("Training complete.")
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    final_validation_loss = evaluate(model, validation_loader, criterion, device)
    print(f"Best model validation loss: {final_validation_loss:.4f}")
    
    
    # Example of text generation with KV cache
    def generate_text(model, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
        """
        Generate text from a prompt using the model.
        
        Args:
            model: The GPT model to use.
            prompt: Text prompt to start generation from.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Temperature for sampling.
            top_k: If set, only sample from the top k most likely tokens.
            
        Returns:
            str: Generated text.
        """
        # Tokenize the prompt
        token_ids = tokenizer.encode(prompt)
        # Convert to tensor and add batch dimension
        x = torch.tensor([token_ids], dtype=torch.long).to(device)
        
        # Generate tokens with KV cache
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(y[0].tolist())
        
        return generated_text
    
    # Test the generation with a prompt
    if os.path.exists(best_model_path):
        model.eval()  # Set model to evaluation mode
        prompt = "Once upon a time"
        print(f"\nPrompt: {prompt}")
        
        # Generate without KV cache for comparison (using the standard forward pass)
        print("\nGenerating without KV cache...")
        import time
        
        # Measure time without KV cache
        start_time = time.time()
        # Create a simple non-cached generation function
        def generate_without_cache(model, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
            model.eval()
            tokens = tokenizer.encode(prompt)
            tokens = torch.tensor([tokens], dtype=torch.long).to(device)
            
            for _ in range(max_new_tokens):
                # Forward pass without caching
                with torch.no_grad():
                    logits, _ = model(tokens, use_cache=False)
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    
                    probs = nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    tokens = torch.cat((tokens, next_token), dim=1)
            
            return tokenizer.decode(tokens[0].tolist())
        
        # Generate without cache
        text_without_cache = generate_without_cache(model, prompt)
        time_without_cache = time.time() - start_time
        
        # Measure time with KV cache
        start_time = time.time()
        text_with_cache = generate_text(model, prompt)
        time_with_cache = time.time() - start_time
        
        # Print results
        print(f"\nGeneration without KV cache took: {time_without_cache:.4f} seconds")
        print(f"Generation with KV cache took: {time_with_cache:.4f} seconds")
        print(f"Speedup factor: {time_without_cache / time_with_cache:.2f}x\n")
        
        print("Generated text (with KV cache):")
        print(text_with_cache)
