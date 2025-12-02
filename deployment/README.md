---
title: Movie Review Sentiment Analysis
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# 🎬 Movie Review Sentiment Analysis

A fine-tuned DistilBERT model for analyzing sentiment in movie reviews, deployed with a beautiful Streamlit interface.

## 🚀 Features

- **Real-time sentiment analysis** for movie reviews
- **Modern UI** with confidence scores and visual indicators
- **Fine-tuned on IMDB dataset** with enhanced modern language understanding
- **Interactive examples** to test different review types
- **Detailed analysis** with confidence levels and metrics

## 🤖 Model

This Space uses a fine-tuned DistilBERT model trained on the IMDB movie review dataset. The model has been enhanced to understand:
- Traditional movie review language
- Modern slang and expressions
- Mixed sentiment reviews
- Short and long-form reviews

## 🎯 How to Use

1. Enter your movie review in the text area
2. Or select one of the example reviews
3. Click "Analyze Sentiment" to see the results
4. View the sentiment (Positive/Negative) with confidence scores

## 📊 Performance

- **Accuracy**: ~93% on IMDB test set
- **Model**: DistilBERT base uncased
- **Training**: Fine-tuned with modern language examples
- **Inference**: Fast CPU/GPU inference with Transformers pipeline

## 🛠️ Technical Stack

- **Model**: Hugging Face Transformers (DistilBERT)
- **Framework**: PyTorch
- **UI**: Gradio
- **Deployment**: Hugging Face Spaces

## 📝 License

Apache 2.0

---

Made with ❤️ using Hugging Face Transformers and Gradio
