"""
Sentiment Analysis Fine-Tuning Script - Modern Slang Edition
============================================================

This script fine-tunes your ALREADY TRAINED model with modern slang data
to improve understanding of contemporary language, emojis, and Gen Z expressions.

Key Steps:
1. Create modern slang dataset
2. Load your existing fine-tuned model
3. Tokenize the modern slang data
4. Configure training arguments (lower LR, fewer epochs)
5. Continue fine-tuning using the Trainer API
6. Evaluate and save the updated model
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n📚 Loading IMDB dataset...")
dataset = load_dataset("stanfordnlp/imdb")

print(f"✅ Dataset loaded!")
print(f"   - Training samples: {len(dataset['train'])}")
print(f"   - Test samples: {len(dataset['test'])}")
print(f"\n🔍 Sample review:")
print(f"   Text: {dataset['train'][0]['text'][:200]}...")
print(f"   Label: {dataset['train'][0]['label']} (0=Negative, 1=Positive)")

# Optional: Use a smaller subset for faster training (for testing purposes)
# Uncomment the lines below to use 10% of the data
# dataset['train'] = dataset['train'].select(range(2500))
# dataset['test'] = dataset['test'].select(range(2500))

# ============================================================================
# 2. LOAD TOKENIZER AND MODEL
# ============================================================================
print("\n🤖 Loading DistilBERT model and tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model for sequence classification (binary: 2 labels)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

print(f"✅ Model loaded: {model_name}")
print(f"   - Parameters: {model.num_parameters():,}")
print(f"   - Labels: {model.config.id2label}")

# ============================================================================
# 3. TOKENIZE DATASET
# ============================================================================
print("\n🔤 Tokenizing dataset...")

def tokenize_function(examples):
    """
    Tokenize text data for the model.
    
    Args:
        examples: Batch of text examples
        
    Returns:
        Tokenized inputs with attention masks
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512  # DistilBERT's maximum sequence length
    )

# Apply tokenization to the entire dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(f"✅ Tokenization complete!")
print(f"   - Example tokenized length: {len(tokenized_datasets['train'][0]['input_ids'])}")

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================================
# 4. DEFINE EVALUATION METRICS
# ============================================================================
def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ============================================================================
# 5. CONFIGURE TRAINING ARGUMENTS
# ============================================================================
print("\n⚙️  Configuring training arguments...")

training_args = TrainingArguments(
    output_dir="./models/sentiment-distilbert-imdb",
    
    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    
    # Evaluation and logging
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=500,
    
    # Performance
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    
    # Misc
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,  # Set to True if you want to push to HF Hub
    report_to="none"  # Disable wandb/tensorboard logging
)

print(f"✅ Training configuration:")
print(f"   - Epochs: {training_args.num_train_epochs}")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Learning rate: {training_args.learning_rate}")
print(f"   - Mixed precision (FP16): {training_args.fp16}")

# ============================================================================
# 6. INITIALIZE TRAINER
# ============================================================================
print("\n🎯 Initializing Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("✅ Trainer initialized!")

# ============================================================================
# 7. TRAIN THE MODEL
# ============================================================================
print("\n🚀 Starting fine-tuning...")
print("=" * 70)

train_result = trainer.train()

print("=" * 70)
print("✅ Training complete!")
print(f"   - Training time: {train_result.metrics['train_runtime']:.2f}s")
print(f"   - Training loss: {train_result.metrics['train_loss']:.4f}")

# ============================================================================
# 8. EVALUATE ON TEST SET
# ============================================================================
print("\n📊 Evaluating on test set...")

eval_results = trainer.evaluate()

print(f"✅ Evaluation results:")
print(f"   - Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"   - Precision: {eval_results['eval_precision']:.4f}")
print(f"   - Recall: {eval_results['eval_recall']:.4f}")
print(f"   - F1 Score: {eval_results['eval_f1']:.4f}")

# ============================================================================
# 9. SAVE THE MODEL
# ============================================================================
print("\n💾 Saving model...")

# Save locally
model_save_path = "./models/sentiment-distilbert-imdb-final"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"✅ Model saved to: {model_save_path}")

# ============================================================================
# 10. PUSH TO HUGGING FACE HUB
# ============================================================================
# To enable: 
# 1. Login: huggingface-cli login
# 2. Set push_to_hub=True in TrainingArguments (line 160)
# 3. Uncomment the code below

# print("\n🌐 Pushing to Hugging Face Hub...")
# trainer.push_to_hub(
#     commit_message="Fine-tuned DistilBERT on IMDB for sentiment analysis"
# )
# print("✅ Model pushed to Hugging Face Hub!")
# print(f"   Access at: https://huggingface.co/YOUR_USERNAME/sentiment-distilbert-imdb")

# ============================================================================
# 11. QUICK INFERENCE TEST
# ============================================================================
print("\n🧪 Testing inference with sample reviews...")

from transformers import pipeline

# Create pipeline for easy inference
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

test_reviews = [
    "This movie was absolutely fantastic! Best film of the year!",
    "Terrible acting and boring plot. Complete waste of time.",
    "The cinematography was beautiful but the story was confusing."
]

for i, review in enumerate(test_reviews, 1):
    result = sentiment_pipeline(review)[0]
    print(f"\n   Test {i}: {review[:60]}...")
    print(f"   Prediction: {result['label']} (confidence: {result['score']:.4f})")

print("\n" + "=" * 70)
print("🎉 Fine-tuning complete! Model ready for deployment.")
print("=" * 70)
