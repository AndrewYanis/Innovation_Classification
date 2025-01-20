from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import json

# Define the model name
model_name = "distilbert-base-uncased"

# Load the datasets
file_path = "/Users/andrew/Inno_Classification/TrainingData.xlsx"
df = pd.read_excel(file_path, sheet_name="TrainingData")  # Main training data
terminology_df = pd.read_excel(file_path, sheet_name="Terminology")  # Terminology tab

# Encode the labels
label_mapping = {label: idx for idx, label in enumerate(df["Innovation Type"].unique())}

# Update label mapping with new types from Terminology tab
new_labels = [label for label in terminology_df["Innovation Type"].unique() if label not in label_mapping]
for label in new_labels:
    label_mapping[label] = len(label_mapping)

# Map labels to dataframes
df["label"] = df["Innovation Type"].map(label_mapping)
terminology_df["label"] = terminology_df["Innovation Type"].map(label_mapping)

# Drop rows with NaN labels in Terminology tab
terminology_df = terminology_df.dropna(subset=["label"])

# Combine datasets
combined_texts = df["Innovation Takeaway"].tolist() + terminology_df["Terminology"].tolist()
combined_labels = df["label"].tolist() + terminology_df["label"].tolist()

# Stratified train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    combined_texts,
    combined_labels,
    test_size=0.2,
    stratify=combined_labels,
    random_state=42
)

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

# Oversample terminologies in the training data
terminology_texts = terminology_df["Terminology"].tolist()
terminology_labels = terminology_df["label"].tolist()

# Combine original training data with duplicated terminologies
augmented_train_texts = train_texts + terminology_texts * 3  # Adjust multiplier as needed
augmented_train_labels = train_labels + terminology_labels * 3

# Recreate the CustomDataset with augmented data
augmented_train_dataset = CustomDataset(augmented_train_texts, augmented_train_labels)
val_dataset = CustomDataset(val_texts, val_labels)

# Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(augmented_train_labels),
    y=augmented_train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# Reset the model
model = DistilBertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(label_mapping)
)

# Update training arguments to include class weights
training_args = TrainingArguments(
    output_dir="./distilbert-finetuned-weighted",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs-weighted",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2
)

# Define a custom loss function to include class weights
def compute_loss(model, inputs):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(model.device))
    loss = loss_fn(logits, labels)
    return loss

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=augmented_train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train and save the model
trainer.train()
model.save_pretrained("./distilbert-finetuned-weighted")
tokenizer.save_pretrained("./distilbert-finetuned-weighted")

# Save the updated label mapping
with open("./distilbert-finetuned-weighted/label_mapping.json", "w") as f:
    json.dump(label_mapping, f)

print("Weighted model training completed and saved.")