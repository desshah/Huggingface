# 🎬 Sentiment Analysis with Hugging Face

A complete end-to-end project demonstrating fine-tuning, deployment, and testing of a sentiment analysis model using Hugging Face tools.

## 📋 Project Overview

This project implements a **binary sentiment classification** system that predicts whether movie reviews express positive or negative sentiment. It showcases the complete ML workflow:

1. **Dataset**: IMDB Movie Reviews (50,000 labeled reviews)
2. **Base Model**: DistilBERT (distilbert-base-uncased)
3. **Fine-Tuning**: Custom training on IMDB dataset
4. **Deployment**: Gradio web interface on Hugging Face Spaces
5. **Testing**: Comprehensive evaluation with diverse test cases

## 🎯 Why These Choices?

### Model: DistilBERT
- **40% smaller** than BERT
- **60% faster** inference
- Retains **97%** of BERT's performance
- Perfect for resource-constrained environments

### Dataset: IMDB
- Industry-standard benchmark
- 50,000 balanced reviews (25k positive, 25k negative)
- Real-world movie review language
- Clear binary classification task

## 🗂️ Project Structure

```
sentiment-analysis-project/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── train_model.py              # Fine-tuning script
├── deployment/
│   ├── app.py                  # Gradio deployment app
│   └── requirements.txt        # Deployment dependencies
├── tests/
│   └── test_model.py           # Test suite with 5+ cases
└── models/                     # Saved model artifacts (local)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-Tune the Model
```bash
python train_model.py
```

This will:
- Load the IMDB dataset
- Tokenize the text data
- Fine-tune DistilBERT for 3 epochs
- Save the model to Hugging Face Hub (optional)

### 3. Test the Model
```bash
python tests/test_model.py
```

### 4. Deploy Locally
```bash
cd deployment
python app.py
```

Visit `http://localhost:7860` to interact with the model.

## 📊 Expected Results

After fine-tuning, the model should achieve:
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~90-93%
- **Inference Speed**: ~50-100ms per review

## 🌐 Deployment on Hugging Face Spaces

1. Create a new Space at https://huggingface.co/spaces
2. Select **Gradio** as the SDK
3. Upload `deployment/app.py` and `deployment/requirements.txt`
4. Your model will be live at: `https://huggingface.co/spaces/deshnaashok/sentiment-analysis`

## 📝 Technical Report

See `REPORT.md` for detailed documentation covering:
- Model architecture and selection rationale
- Fine-tuning methodology and hyperparameters
- Deployment architecture
- Comprehensive test results and analysis

## 🧪 Test Cases

The test suite includes:
1. **Clear Positive**: Unambiguous positive sentiment
2. **Clear Negative**: Unambiguous negative sentiment
3. **Mixed/Subtle**: Nuanced language with conflicting sentiments
4. **Out-of-Domain**: Non-review text
5. **Slang/Informal**: Modern internet language and emojis

## 📚 Key Learnings

- **Tokenization** is crucial for model performance
- **Fine-tuning** adapts general language understanding to specific tasks
- **DistilBERT** offers excellent speed/accuracy trade-off
- **Gradio** makes deployment incredibly simple

## 🔗 Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [IMDB Dataset Card](https://huggingface.co/datasets/stanfordnlp/imdb)
- [DistilBERT Model Card](https://huggingface.co/distilbert-base-uncased)

## 📜 License

MIT License - Feel free to use this project for learning and development.

---

**Built with ❤️ using Hugging Face 🤗**
