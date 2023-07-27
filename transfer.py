import torch
from transformers import AutoModelForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, AutoTokenizer, TrainerCallback
from torch.utils.data import Dataset

theogony_model = AutoModelForSequenceClassification.from_pretrained("./train-hesiod/theogony_model")
works_model = AutoModelForSequenceClassification.from_pretrained("./worksandday/works_model")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token
model = GPT2LMHeadModel.from_pretrained("gpt2")

with open("books.txt", "r") as file:
    text = file.read()

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=300,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

def split_text_into_chunks(text, max_length=512):
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = start_idx + max_length
        if end_idx >= len(text):
            end_idx = len(text)
        chunks.append(text[start_idx:end_idx])
        start_idx += max_length
    return chunks

def check_input_ids_within_range(input_ids, tokenizer):
    vocab_size = tokenizer.vocab_size
    for id in input_ids.flatten():
        if id < 0 or id >= vocab_size:
            return False
    return True

class TextDataset(Dataset):
    def __init__(self, tokenizer, text_chunks, max_length):
        self.examples = []
        for chunk in text_chunks:
            encoding = tokenizer.encode_plus(chunk, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            if check_input_ids_within_range(encoding['input_ids'], tokenizer):
                self.examples.append({
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Split the text into smaller chunks
text_chunks = split_text_into_chunks(text)

# Create a list of datasets for theogony_model and works_model
theogony_datasets = [TextDataset(tokenizer, text_chunks, max_length=512) for _ in range(training_args.num_train_epochs)]
works_datasets = [TextDataset(tokenizer, text_chunks, max_length=512) for _ in range(training_args.num_train_epochs)]

# Define the loss callback
class LossCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.losses = []

    def on_step_end(self, args, state, control, **kwargs):
        if 'loss' in state.log_history[-1]:
            loss = state.log_history[-1]['loss']
            step = state.global_step
            print(f"Step {step}: Loss - {loss:.4f}")
            self.losses.append(loss)

    def on_train_end(self, args, state, control, **kwargs):
        print('Training loss:', self.losses)

def transfer_learning(model, train_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[LossCallback()]
    )
    
    trainer.train()
    return model, trainer.state.log_history

print("Fine-tuning theogony_model")
theogony_train_losses = []
for i, dataset in enumerate(theogony_datasets):
    if len(dataset) > 0:
        theogony_model, loss = transfer_learning(theogony_model, dataset)
        theogony_train_losses.extend(loss)

print("Fine-tuning works_model")
works_train_losses = []
for i, dataset in enumerate(works_datasets):
    if len(dataset) > 0:
        works_model, loss = transfer_learning(works_model, dataset)
        works_train_losses.extend(loss)

# Save the fine-tuned models
theogony_model.save_pretrained("theogony_model_transfer")
works_model.save_pretrained("works_model_transfer")
