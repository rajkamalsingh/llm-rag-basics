from transformers import AutoTokenizer

def get_tokenizer(model_name="google/flan-t5-base"):
    return AutoTokenizer.from_pretrained(model_name)

def preprocess(example):
    """
    Preprocess squad entry into input/output form for t5-base.
    input: question + context
    output: answer (first answer if multiple)
    :param model_name:
    :return:
    """

    question = example["question"]
    context = example["context"]
    #take first answer if multiple
    answer = example["answers"]["text"][0] if len(example["answers"]["text"]) >0 else ""

    return {
        "input_text":f"Question: {question}\n Context: {context}\nAnswer:",
        "target_text":answer
    }

def tokenize_function(batch, tokenizer, max_length=256):
    """
    tokenize both input and target sequence.
    :param batch:
    :param tokenizer:
    :param max_length:
    :return:
    """
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=max_length)
    labels = tokenizer(batch["target_text"], padding="max_length", truncation=True, max_length=max_length)

    inputs["labels"]=labels["input_ids"]
    return inputs

