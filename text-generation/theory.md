# Theory: Text Generation with LLMs

## What happens in this project?
- We load GPT-2, a pretrained Transformer-based language model.
- The input text is tokenized (converted into token IDs).
- The model generates next tokens iteratively.
- The output tokens are decoded back to text.

## Key Concepts
- **Tokenization**: Breaking input into subword tokens.
- **Max length**: The total length of input + generated tokens.
- **Greedy vs sampling**: This example uses default (greedy) decoding.

## Tools
- Hugging Face `pipeline` API
- Model: `gpt2`