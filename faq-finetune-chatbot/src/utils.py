from os import truncate

from transformers import AutoTokenizer
def get_tokenizer(model_name="google/flan-t5-small"):
    return AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    return {
        "input_text":"Question: " +example["query"] +"\nAnswer:",
        "target_text": "This is a placeholder answer."
    }

def tokenize_function(batch, tokenizer, max_length=128):
    print(batch["input_text"][:2])
    print(batch["target_text"][:2])
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation = True, max_length=max_length)
    labels = tokenizer(batch["target_text"], padding="max_length", truncation = True, max_length = max_length)
    inputs["labels"] = labels["input_ids"]

    return inputs

