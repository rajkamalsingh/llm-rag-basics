# Project 1: Theory — Text Generation with GPT-2 (Hugging Face)

---

## What is an LLM?

A **Large Language Model (LLM)** is a deep neural network trained on massive text corpora to predict the next token (word or subword) in a sequence.  
Key features of LLMs:
- Based on the **Transformer** architecture.
- Pre-trained on billions of tokens (books, websites, code, etc.).
- Can perform many tasks with no or minimal fine-tuning (text generation, summarization, translation, etc.).

**GPT-2**, used in this project:
- Released by OpenAI (2019).
- 1.5 billion parameters (we’re using the base model with ~117M for faster local inference).
- Pre-trained on web text (without supervised fine-tuning).

---

## Key Concepts

### Tokenization
- Converts raw text into tokens (subword units).
- Example: `"happiness"` → `["hap", "piness"]`
- The model processes token IDs, not raw text.

---

### Text Generation (Language Modeling)
- **Causal Language Model**: Predicts next token based on previous tokens.
- Repeats this step to extend text.
- Each generated token is fed back as input for the next prediction.

---

### Decoding Strategies
When generating text, the model produces a probability distribution over possible next tokens.  
How we choose the next token affects the output style:
- **Greedy Decoding**: Always pick the most probable token. Can be repetitive or dull.
- **Sampling**: Randomly pick tokens according to their probability.
- **Top-k Sampling**: Pick from top *k* tokens (we aren’t using this in Project 1).
- **Top-p (nucleus sampling)**: Pick from the smallest set of tokens whose probabilities sum to *p*.
- **Temperature**: Controls randomness. Higher = more creative, lower = more deterministic.

Example in our code:
```python
output = generator(
    prompt,
    max_length=100,
    temperature=0.7,    # Adds randomness
    top_p=0.9,          # Use nucleus sampling
    do_sample=True      # Enable sampling (not greedy)
)
```
---
### The Hugging Face pipeline
- Provides a high-level API for common tasks (text-generation, summarization, etc.).
- Handles tokenization, model inference, decoding, and output formatting
---
### Summary of Parameters
| Parameter     | Meaning                                                    |
| ------------- | ---------------------------------------------------------- |
| `max_length`  | Total length of input + generated tokens                   |
| `temperature` | Randomness in token selection                              |
| `top_p`       | Nucleus sampling: select tokens adding up to p probability |
| `do_sample`   | Enables random sampling (instead of always most likely)    |
---
## How GPT-2 Works (Simplified)
```
Input prompt → Tokenizer → Transformer layers → Next-token prediction → Decoder → Output text
```
- The Transformer layers apply self-attention to model dependencies in the sequence.
- The model outputs one token at a time, updating the input with each new token.
---

### Why use GPT-2 in this project?
- Runs locally without API.
- Small enough to run on CPU (base model ~117M parameters).
- Easy entry point to learn generation concepts.

#### Alternatives to GPT-2
- GPT-Neo / GPT-J (EleutherAI)
- Mistral 7B (requires GPU / quantization)
- OpenAI GPT-3.5 (API-based)