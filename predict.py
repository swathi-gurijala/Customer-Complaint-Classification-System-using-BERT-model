import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Load the trained model and tokenizer
model_path = "./complaint_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load category mapping
with open("category_mapping.json", "r") as f:
    category_mapping = json.load(f)
category_mapping = {int(v): k for k, v in category_mapping.items()}  # Reverse mapping

# Function to predict complaint category
def predict_category(complaint):
    inputs = tokenizer(complaint, truncation=True, padding='max_length', max_length=256, return_tensors="pt")
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_category = category_mapping[predicted_class]
    
    return predicted_category

# Test the function
if __name__ == "__main__":
    sample_complaint = input("Enter a complaint: ")
    predicted_category = predict_category(sample_complaint)
    print(f"Predicted Category: {predicted_category}")
