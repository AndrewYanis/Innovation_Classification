from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
from sklearn.metrics import classification_report

# Load the trained model and tokenizer
model_path = "./distilbert-finetuned-weighted"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load the label mapping
with open(f"{model_path}/label_mapping.json", "r") as f:
    label_mapping = json.load(f)

# Reverse label mapping for easier interpretation
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Prepare test texts (replace with actual test dataset)
test_texts = [
    "This innovation focuses on improving agricultural yields.",
    "The project builds access to market.",
    "This innovation is a financial platform that helps farmers."
]

# Tokenize the test texts
inputs = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

# Put the model in evaluation mode
model.eval()

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).tolist()

# Map predictions back to label names
predicted_labels = [reverse_label_mapping[label] for label in predictions]

# Display results
for text, label in zip(test_texts, predicted_labels):
    print(f"Text: {text}\nPredicted Label: {label}\n")

# Optional: Evaluate on a labeled test dataset
# Example true labels for the test_texts (replace with actual labels)
true_labels = list(label_mapping.keys())[:len(test_texts)]  # Replace with actual labels if known

# Convert true labels to indices
try:
    true_labels_idx = [label_mapping[label] for label in true_labels]
except KeyError as e:
    print(f"Invalid label: {e}. Please check the true_labels list and ensure it matches label_mapping.")
    true_labels_idx = []

# Only proceed if true_labels_idx is valid
if true_labels_idx:
    # Extract unique classes from predictions and true labels
    unique_classes = sorted(set(true_labels_idx + predictions))

    # Generate a classification report
    print(classification_report(
        true_labels_idx,
        predictions,
        labels=unique_classes,
        target_names=[reverse_label_mapping[c] for c in unique_classes]
    ))