# FAQ Chatbot Fine-Tuning with FLAN-T5

This project fine-tunes the `google/flan-t5-small` model on a custom FAQ dataset to create a simple chatbot.  
It uses the Hugging Face `datasets` and `transformers` libraries.

---

## Project Structure
```faq-finetune-chatbot/
│── src/
│ ├── train.py # Training script
│ ├── utils.py # Preprocessing & tokenization utilities
│── logs/ # Training logs
│── faq_model/ # Saved fine-tuned model
│── README.md # Project documentation
```
---
##  Requirements

Install dependencies before running:

```bash
pip install -u requirements.txt
```
---

## Training

Run the training script:
```bash
python src/train.py
```
This will:
- Load the dataset (farzanrahmani/chatbot-FAQ-queries from Hugging Face Hub).
- Preprocess the data into input-output pairs.
- Tokenize using the FLAN-T5 tokenizer.
- Train the model with Seq2SeqTrainer.
- Save the trained model in faq_model/
---
## Key Files
### train.py
- Loads dataset and applies preprocessing.
- Sets up tokenizer & model (**flan-t5-small**).
- Defines training arguments and runs fine-tuning.
- Saves the trained model.

### utils.py
- get_tokenizer() → Loads the tokenizer.
- preprocess() → Formats dataset into prompt + placeholder answer.
- tokenize_function() → Tokenizes inputs and labels.

---
## Training Configuration
- Model: google/flan-t5-small
- Batch Size: 4
- Epochs: 3
- Learning Rate: 2e-5
- Max Seq Length: 128

---
## Training Output
- Model checkpoints & final model: faq_model/
- Training logs: logs/

---

## Next Steps (Scaling Up)
- Replace placeholder answers with real answers from dataset.
- Train with a larger model (e.g., flan-t5-base).
- Experiment with more epochs & different learning rates.
- Add evaluation metrics like **BLEU, ROUGE, or F1**.
- Deploy the fine-tuned model via FastAPI or Gradio for interactive chatbot usage.

---

### Notes
- Current version uses dummy placeholder answers.
- For real chatbot usage, dataset should include actual FAQ answers.
- Ensure GPU runtime (CUDA) for faster training.