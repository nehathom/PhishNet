# train_distilbert.py

import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import evaluate 


def load_urls(benign_path, malware_path):
    with open(benign_path, 'r') as f:
        benign_urls = [line.strip() for line in f if line.strip()]
    with open(malware_path, 'r') as f:
        malware_urls = [line.strip() for line in f if line.strip()]
    return benign_urls, malware_urls

def main():
    benign_urls, malware_urls = load_urls('/Users/nehathomas/Desktop/FishNet/data/Train/benign_Train.txt', '/Users/nehathomas/Desktop/FishNet/data/Train/malign_Train.txt')

    # Create DataFrame
    df = pd.DataFrame({
        'url': benign_urls + malware_urls,
        'label': [0]*len(benign_urls) + [1]*len(malware_urls)
    })

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Tokenize function
    def tokenize_fn(examples):
        return tokenizer(examples['url'], truncation=True, padding='max_length', max_length=128)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    # Split into train/test
    split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = split['train']
    eval_dataset = split['test']

    # Load accuracy metric
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train!
    trainer.train()

if __name__ == "__main__":
    main()