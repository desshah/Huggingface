"""
Quick Start Guide - Sentiment Analysis Project
===============================================

This script provides a quick demonstration of loading and using the model.
"""

from transformers import pipeline

# Load the sentiment analysis pipeline
print("Loading model...")
sentiment = pipeline("sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english")

# Test examples
examples = [
    "This movie was absolutely fantastic!",
    "Terrible waste of time.",
    "It was okay, nothing special."
]

print("\n🎬 Sentiment Analysis Demo\n" + "="*50)

for text in examples:
    result = sentiment(text)[0]
    print(f"\nText: {text}")
    print(f"→ {result['label']}: {result['score']:.2%} confidence")

print("\n" + "="*50)
print("✅ Demo complete! See train_model.py for fine-tuning.")
