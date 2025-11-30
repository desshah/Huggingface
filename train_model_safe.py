"""
Safe Training Script for Sentiment Analysis
===========================================

Optimized for Mac/CPU training with memory management and progress tracking.
This script uses a smaller dataset by default for quick testing.
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

print("=" * 70)
print("🎬 SENTIMENT ANALYSIS - SAFE TRAINING MODE")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Training mode: 'test' (small subset) or 'full' (complete dataset)
TRAINING_MODE = 'test'  # Change to 'full' for production training
SMALL_SUBSET_SIZE = 2500  # For quick testing
BATCH_SIZE = 8  # Smaller batch for safety
NUM_EPOCHS = 3

print(f"\n⚙️  Configuration:")
print(f"   Mode: {TRAINING_MODE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Epochs: {NUM_EPOCHS}")

# Detect device
if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
    print(f"   Device: Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"   Device: CUDA GPU")
else:
    device = "cpu"
    print(f"   Device: CPU")

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
print("\n📚 Loading IMDB dataset from Stanford NLP...")
print("   (This may take a few minutes on first run)")

try:
    dataset = load_dataset("stanfordnlp/imdb")
    print(f"✅ Dataset loaded!")
    print(f"   - Training samples: {len(dataset['train'])}")
    print(f"   - Test samples: {len(dataset['test'])}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("   Trying alternative: 'imdb'")
    dataset = load_dataset("imdb")

# Use smaller subset for testing
if TRAINING_MODE == 'test':
    print(f"\n⚠️  TEST MODE: Using {SMALL_SUBSET_SIZE} samples for quick training")
    dataset['train'] = dataset['train'].select(range(SMALL_SUBSET_SIZE))
    dataset['test'] = dataset['test'].select(range(min(SMALL_SUBSET_SIZE, len(dataset['test']))))
    print(f"   - Training samples: {len(dataset['train'])}")
    print(f"   - Test samples: {len(dataset['test'])}")

print(f"\n🔍 Sample review:")
print(f"   Text: {dataset['train'][0]['text'][:150]}...")
print(f"   Label: {dataset['train'][0]['label']} (0=Negative, 1=Positive)")

# ============================================================================
# 2. LOAD TOKENIZER AND MODEL
# ============================================================================
print("\n🤖 Loading DistilBERT model and tokenizer...")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

print(f"✅ Model loaded: {model_name}")
print(f"   - Parameters: {model.num_parameters():,}")

# ============================================================================
# 3. TOKENIZE DATASET
# ============================================================================
print("\n🔤 Tokenizing dataset...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256  # Reduced from 512 for faster training
    )

tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=dataset['train'].column_names
)

print(f"✅ Tokenization complete!")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================================
# 4. EVALUATION METRICS
# ============================================================================
def compute_metrics(eval_pred):
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
# 5. TRAINING ARGUMENTS
# ============================================================================
print("\n⚙️  Configuring training arguments...")

# Estimate training time
samples_per_epoch = len(dataset['train'])
steps_per_epoch = samples_per_epoch // BATCH_SIZE
total_steps = steps_per_epoch * NUM_EPOCHS

if device == "cpu":
    estimated_time_min = (total_steps * 2) / 60  # ~2 sec per step on CPU
elif device == "mps":
    estimated_time_min = (total_steps * 0.5) / 60  # ~0.5 sec per step on MPS
else:
    estimated_time_min = (total_steps * 0.2) / 60  # ~0.2 sec per step on GPU

print(f"   Total training steps: {total_steps}")
print(f"   Estimated time: {estimated_time_min:.1f} minutes")

training_args = TrainingArguments(
    output_dir="./models/sentiment-distilbert-imdb",
    
    # Training hyperparameters
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=2e-5,
    weight_decay=0.01,
    
    # Evaluation and logging
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    
    # Performance
    fp16=False,  # Disabled for Mac compatibility
    
    # Safety
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to="none",
    
    # Prevent memory issues
    dataloader_num_workers=0,  # Avoid multiprocessing issues
    gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
)

print("✅ Training configuration ready!")

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
print("\n" + "=" * 70)
print("🚀 Starting fine-tuning...")
print("=" * 70)
print(f"⏱️  Estimated time: {estimated_time_min:.1f} minutes")
print(f"📊 Progress will be logged every 100 steps")
print("=" * 70 + "\n")

try:
    train_result = trainer.train()
    
    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"   - Training time: {train_result.metrics['train_runtime']:.2f}s ({train_result.metrics['train_runtime']/60:.2f} minutes)")
    print(f"   - Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # ============================================================================
    # 8. EVALUATE
    # ============================================================================
    print("\n📊 Evaluating on test set...")
    
    eval_results = trainer.evaluate()
    
    print(f"✅ Evaluation results:")
    print(f"   - Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
    print(f"   - Precision: {eval_results['eval_precision']:.4f}")
    print(f"   - Recall: {eval_results['eval_recall']:.4f}")
    print(f"   - F1 Score: {eval_results['eval_f1']:.4f}")
    
    # ============================================================================
    # 9. SAVE THE MODEL
    # ============================================================================
    print("\n💾 Saving model...")
    
    model_save_path = "./models/sentiment-distilbert-imdb-final"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    print(f"✅ Model saved to: {model_save_path}")
    
    # ============================================================================
    # 10. QUICK TEST
    # ============================================================================
    print("\n🧪 Testing inference...")
    
    from transformers import pipeline
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    
    test_reviews = [
        "This movie was absolutely fantastic!",
        "Terrible waste of time.",
    ]
    
    for review in test_reviews:
        result = sentiment_pipeline(review)[0]
        print(f"\n   '{review}'")
        print(f"   → {result['label']}: {result['score']:.4f}")
    
    print("\n" + "=" * 70)
    print("🎉 SUCCESS! Model is ready for deployment.")
    print("=" * 70)
    
    if TRAINING_MODE == 'test':
        print("\n💡 TIP: To train on full dataset, change TRAINING_MODE to 'full'")
        print("   in train_model_safe.py (line 33) and run again.")
    
    print("\n📋 Next Steps:")
    print("   1. Test the model: python tests/test_model.py")
    print("   2. Deploy locally: cd deployment && python app.py")
    print("   3. Read the report: cat REPORT.md")
    
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ Training failed!")
    print("=" * 70)
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check if you have enough RAM (8GB+ recommended)")
    print("2. Try reducing BATCH_SIZE to 4 or 2")
    print("3. Make sure you're in the virtual environment")
    print("4. Check if all dependencies are installed: pip install -r requirements.txt")
    raise
