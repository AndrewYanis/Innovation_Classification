# README

## Overview
This repository contains the code for training and testing a fine-tuned DistilBERT model for text classification. The project leverages the `transformers` library by Hugging Face and includes:

1. **Training the Weighted Model**: A script to fine-tune a pre-trained DistilBERT model using a weighted dataset to emphasize specific terms and labels.
2. **Testing the Trained Model**: A script to evaluate the trained model by making predictions on test data and generating evaluation metrics.

## File Structure

```
.
├── TrainingData.xlsx         # Input data containing training and terminology sheets
├── Weight_Training.py        # Script for training the model with weighted dataset
├── Model_Testing.py          # Script for testing the trained model
├── distilbert-finetuned-weighted/ # Directory where the fine-tuned model is saved
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── label_mapping.json
└── README.md                 # This documentation file
```

## Requirements

Install the required Python packages using the following command:

```bash
pip install transformers scikit-learn torch pandas
```

Ensure you have the necessary dataset file `TrainingData.xlsx` in the root directory.

## Training the Model

### Input Data
- **TrainingData.xlsx**:
  - `TrainingData` sheet: Contains the main training dataset with innovation takeaways and their corresponding types.
  - `Terminology` sheet: Contains terminology-specific entries to emphasize during training.

### Script: `Weight_Training.py`
1. Loads and preprocesses the training data.
2. Encodes labels and augments the dataset to emphasize terminologies.
3. Fine-tunes a pre-trained DistilBERT model.
4. Saves the fine-tuned model and label mapping for later use.

#### Run the Training Script
```bash
python Weight_Training.py
```

The trained model will be saved in the `distilbert-finetuned-weighted` directory.

## Testing the Model

### Script: `Model_Testing.py`
1. Loads the fine-tuned model and tokenizer from the `distilbert-finetuned-weighted` directory.
2. Accepts test input texts and performs predictions.
3. Optionally, evaluates the predictions against true labels using classification metrics.

#### Run the Testing Script
```bash
python Model_Testing.py
```

### Output
- Predictions for the test texts.
- Classification metrics (e.g., precision, recall, F1-score) if true labels are provided.

## Notes
- Ensure the `distilbert-finetuned-weighted` directory exists and contains the fine-tuned model files before running the testing script.
- Replace placeholder test texts and labels in `Model_Testing.py` with your actual test data for accurate results.

## Author
Andrew Yan

