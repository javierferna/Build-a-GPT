# 🤖 Build-a-GPT

This  repository provides an educational implementation of a GPT-style autoregressive Transformer model trained from scratch. It enables users to build, train, and generate text using custom documents. In this case, I used the classic novel *"Twenty Thousand Leagues Under the Sea"* by Jules Verne.

---

**Key features:**

* 📚 **GPT Transformer Model:** An autoregressive Transformer decoder architecture inspired by GPT.
* 🔄 **Rotary Positional Embeddings (RoPE):** Implements SOTA positional encodings to effectively capture relative token positions.
* 🚀 **Text Generation Tools:** Generate original text using adjustable parameters such as temperature (controls randomness) and top-k sampling (limits selection to k highest-probability tokens).
* ⚡ **KV Caching:** Efficient text generation using key-value caching to reduce redundant computations.
* 📖 **Custom Training Data:** Easily train your GPT model on custom input text (`input.txt` provided by default from Jules Verne's classic).

---

Scripts Included:

`my_gpt.py` - Implementation of the GPT Transformer architecture, including training and evaluation.  
`generate.py` - Script to generate new text based on trained model weights. Supports temperature and top-k adjustments.  
`input.txt` - Default training corpus sourced from *"Twenty Thousand Leagues Under the Sea"* by Jules Verne.  

---

Usage Example:

```example bash
python my_gpt.py --file_path input.txt --batch_size 32 --block_size 128 --num_epochs 4
python generate.py --temperature 1.0 --top_k 40
