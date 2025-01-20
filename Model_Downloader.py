from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Specify the correct model name
model_name = "distilbert-base-uncased"

# Download and save the model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# Save locally
tokenizer.save_pretrained("./distilbert-base-uncased-tokenizer")
model.save_pretrained("./distilbert-base-uncased-model")