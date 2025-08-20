from transformers import pipeline
from utils import get_tokenizer

def main():
    model_dir="../faq_model"
    tokenizer = get_tokenizer()

    faq_pipe = pipeline("text2text-generation", model = model_dir, tokenizer = tokenizer)

    # test with a sample query
    query = "Question: What is AI?\nAnswer:"
    response = faq_pipe(query, max_length = 100, clean_up_tokenization_spaces = True)
    print("Q: what is AI?")
    print("A:", response[0]["generated_text"])

if __name__ =="__main__":
    main()