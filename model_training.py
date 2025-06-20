import pandas as pd
import torch
import evaluate
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import json

# Load dataset
df = pd.read_csv("customer_complaints.csv")
df = df[['complaint', 'category']]  # Ensure these columns exist

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode data
class ComplaintDataset(Dataset):
    def __init__(self, complaints, labels):
        self.complaints = complaints
        self.labels = labels

    def __len__(self):
        return len(self.complaints)

    def __getitem__(self, idx):
        encoding = tokenizer(self.complaints[idx], truncation=True, padding='max_length', max_length=256, return_tensors="pt")
        encoding = {key: val.squeeze() for key, val in encoding.items()}
        encoding["labels"] = torch.tensor(self.labels[idx])
        return encoding

# Convert categories to numbers
category_mapping = {category: idx for idx, category in enumerate(df['category'].unique())}
df['category_id'] = df['category'].map(category_mapping)

# Save category mapping
with open("category_mapping.json", "w") as f:
    json.dump(category_mapping, f)

# Split dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(df['complaint'].tolist(), df['category_id'].tolist(), test_size=0.2)

# Create datasets
train_dataset = ComplaintDataset(train_texts, train_labels)
test_dataset = ComplaintDataset(test_texts, test_labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(category_mapping))

# Training arguments
training_args = TrainingArguments(
    output_dir="./complaint_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=30,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    save_total_limit=2   
)

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load accuracy metric (updated)
accuracy_metric = evaluate.load("accuracy")

# Function to compute accuracy (updated)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics  # Compute accuracy
)

# Train model
trainer.train()

# Evaluate model
eval_results = trainer.evaluate()
print(f"✅ Evaluation Accuracy: {eval_results['eval_accuracy']:.2%}")  # Print accuracy in percentage

# Save model and tokenizer
model.save_pretrained("./complaint_model")
tokenizer.save_pretrained("./complaint_model")

print("🎉 Training completed successfully!")
