from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils import preprocess, tokenize_function, get_tokenizer

def main():
    # load dataset
    dataset = load_dataset("farzanrahmani/chatbot-FAQ-queries")

    #preprocess dataset
    train_dataset = dataset["train"].map(preprocess)

    # Tokenizer and model

    model_name = "google/flan-t5-small"
    tokenizer = get_tokenizer(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # tokenize
    tokenized_dataset = train_dataset.map(
        lambda batch : tokenize_function(batch, tokenizer),
        batched = True
    )

    # training args
    training_args = Seq2SeqTrainingArguments(
        output_dir= "../faq_model",
        evaluation_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size= 4,
        num_train_epochs=3,
        save_strategy = "epoch",
        looging_dir = "../logs",
        push_to_hub= False

    )
    trainer = Seq2SeqTrainer(
        model = model,
        arg=training_args,
        train_dataset= tokenized_dataset,
        tokenizer = tokenizer
    )

    trainer.train()

    #save model
    trainer.save_model("../faq_model")


if __name__ == "__main__":
    main()