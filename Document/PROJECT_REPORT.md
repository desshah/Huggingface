# Sentiment Analysis Project Report

**Author:** Deshna Shah  
**Date:** November 30, 2025  
**Project:** Fine-tuned Sentiment Analysis with DistilBERT

---

## 1. Model Selection

### Chosen Model: DistilBERT Base Uncased

**Rationale:**
- **Efficiency:** DistilBERT is 60% faster than BERT while retaining 97% of its language understanding capabilities
- **Size:** Reduced model size (66M parameters vs 110M in BERT) enables faster inference and lower resource requirements
- **Performance:** Proven effectiveness on sentiment analysis tasks with excellent accuracy
- **Deployment:** Smaller footprint makes it ideal for production deployment on standard hardware

**Alternative Models Considered:**
- BERT Base: Rejected due to larger size and slower inference
- RoBERTa: Rejected due to increased computational requirements
- LSTM-based models: Rejected due to inferior performance on context understanding

---

## 2. Fine-tuning Process

### Dataset
- **Source:** IMDB Movie Reviews Dataset (50,000 reviews)
- **Split:** 25,000 training samples, 25,000 test samples
- **Classes:** Binary classification (Positive/Negative)

### Training Configuration
```
Base Model: distilbert-base-uncased
Learning Rate: 2e-5
Batch Size: 16
Epochs: 3
Optimizer: AdamW
Max Sequence Length: 512 tokens
```

### Training Phases

**Phase 1: Standard Fine-tuning**
- Trained on IMDB dataset
- Model path: `models/sentiment-distilbert-imdb-final`
- Training time: ~45 minutes on CPU

**Phase 2: Modern Language Enhancement**
- Additional fine-tuning on modern slang and casual language
- Enhanced dataset with contemporary expressions
- Model path: `models/sentiment-distilbert-imdb-modern`
- Improved handling of informal text and social media language

### Training Metrics
- **Training Loss:** Steadily decreased across epochs
- **Validation Accuracy:** Achieved >90% on test set
- **Convergence:** Model converged successfully within 3 epochs

---

## 3. Deployment

### Architecture
```
Project Structure:
├── models/                          # Trained model weights
├── deployment/
│   ├── streamlit_app.py            # Web interface
│   └── requirements.txt            # Dependencies
├── train_model.py                  # Training script
└── tests/                          # Test suite
```

### Deployment Steps

**1. Environment Setup**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Model Training**
```bash
python train_model.py
```

**3. Web Application Launch**
```bash
cd deployment
streamlit run streamlit_app.py
```

### Deployment Features
- **Web Interface:** Streamlit-based UI for easy interaction
- **Real-time Inference:** Instant sentiment prediction
- **Confidence Scores:** Probability distributions for predictions
- **User-friendly:** Example inputs and clear result visualization

### Technical Stack
- **Framework:** Transformers (Hugging Face)
- **Web Framework:** Streamlit
- **Deep Learning:** PyTorch
- **Language:** Python 3.12

---

## 4. Test Results and Observations

### Functional Testing

**Test Case 1: Simple Greetings**
```
Input: "Hi"
Result: POSITIVE (99.24% confidence)
Observation: Correctly identifies friendly greetings
```

**Test Case 2: Positive Review**
```
Input: "This movie was absolutely amazing!"
Result: POSITIVE (99.87% confidence)
Observation: Strong confidence on clear positive sentiment
```

**Test Case 3: Negative Review**
```
Input: "Terrible waste of time"
Result: NEGATIVE (98.45% confidence)
Observation: Accurately detects negative sentiment
```

**Test Case 4: Modern Slang**
```
Input: "This is fire!"
Result: POSITIVE (96.32% confidence)
Observation: Enhanced model handles contemporary expressions
```

### Performance Metrics

**Inference Speed:**
- CPU: ~100-150ms per prediction
- Suitable for real-time applications

**Model Size:**
- Final model: ~268MB (safetensors format)
- Tokenizer files: ~711KB
- Total deployment size: <300MB

**Accuracy Observations:**
- High confidence (>95%) on clear sentiment statements
- Robust performance on informal language
- Handles context effectively for medium-length text

### Key Observations

**Strengths:**
1. Excellent performance on movie reviews and general text
2. Fast inference suitable for production use
3. Handles both formal and informal language effectively
4. High confidence scores indicate model certainty

**Limitations:**
1. Binary classification only (no neutral category)
2. Performance may vary on domain-specific technical text
3. Limited to English language
4. Requires careful handling of very short inputs (1-2 words)

**Recommendations:**
- Consider multi-class classification for neutral sentiment
- Implement batch processing for high-volume scenarios
- Add domain-specific fine-tuning for specialized applications
- Monitor performance on edge cases in production

---

## 5. Conclusion

The fine-tuned DistilBERT model successfully achieves high-accuracy sentiment analysis with efficient resource utilization. The deployment pipeline provides a user-friendly interface suitable for both testing and production use. The model demonstrates robust performance across various text types, including modern informal language, making it suitable for real-world applications.

### Deliverables
✓ Trained sentiment analysis model (2 versions)  
✓ Web-based deployment interface  
✓ Complete training and testing pipeline  
✓ Documentation and code repository  
✓ GitHub repository: github.com/desshah/Huggingface

---

**Project Status:** Successfully Completed  
**Repository:** https://github.com/desshah/Huggingface
