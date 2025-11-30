"""
Streamlit Deployment App for Sentiment Analysis
================================================

A beautiful web interface for the fine-tuned sentiment analysis model.

Features:
- Clean, modern UI with Streamlit
- Real-time sentiment predictions
- Confidence scores with visual indicators
- Example inputs for quick testing
- Emoji indicators for sentiment
"""

import streamlit as st
from transformers import pipeline
import torch
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    """Load the sentiment analysis model (cached for performance)"""
    # Use the modern slang-enhanced model!
    model_path = "../models/sentiment-distilbert-imdb-modern"
    
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        return sentiment_pipeline, model_path
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.title("🎬 Movie Review Sentiment Analysis ")
    st.markdown("---")
    
    # Load model
    with st.spinner("🔄 Loading model..."):
        sentiment_pipeline, model_path = load_model()
    
    if sentiment_pipeline is None:
        st.stop()
    
    # Main input area
    st.subheader("📝 Enter Your Review")
    
    # Example reviews
    examples = {
        "Select an example...": "",
        "🎭 Positive: Masterpiece": "This movie was an absolute masterpiece! The best I've seen all year.",
        "🎭 Negative: Disappointing": "A completely worthless script and terrible acting. Save your money.",
        "🎭 Modern Slang: Fire": "OMG this film was fire 🔥. So good.",
        "🎭 Mixed Sentiment": "Great acting saved an otherwise mediocre story. The visuals were stunning but couldn't make up for the weak plot.",
        "🎭 Lukewarm": "Not the best movie ever, but certainly not the worst. Perfectly fine for a rainy afternoon.",
    }
    
    selected_example = st.selectbox("Or choose an example:", list(examples.keys()))
    
    # Text input
    default_text = examples[selected_example] if selected_example != "Select an example..." else ""
    user_input = st.text_area(
        "Type or paste your movie review here:",
        value=default_text,
        height=150,
        placeholder="e.g., This movie was amazing! The acting was superb and the story kept me engaged throughout..."
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
    
    # Perform analysis
    if analyze_button or (user_input and user_input != ""):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🤔 Analyzing sentiment..."):
                try:
                    # Get prediction
                    result = sentiment_pipeline(user_input[:512])[0]  # Limit to 512 tokens
                    label = result['label']
                    confidence = result['score']
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Sentiment with emoji
                    if label == "POSITIVE":
                        sentiment_emoji = "😊"
                        sentiment_color = "green"
                    else:
                        sentiment_emoji = "😞"
                        sentiment_color = "red"
                    
                    # Confidence level indicator
                    if confidence > 0.8:
                        conf_indicator = "🟢"
                        conf_level = "High"
                    elif confidence > 0.6:
                        conf_indicator = "🟡"
                        conf_level = "Medium"
                    else:
                        conf_indicator = "🔴"
                        conf_level = "Low"
                    
                    # Display in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {sentiment_emoji} **Sentiment**")
                        st.markdown(f":{sentiment_color}[**{label}**]")
                    
                    with col2:
                        st.markdown(f"### {conf_indicator} **Confidence**")
                        st.markdown(f"**{confidence:.1%}** ({conf_level})")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Additional info
                    with st.expander("📈 Detailed Analysis"):
                        st.json({
                            "sentiment": label,
                            "confidence_score": round(confidence, 4),
                            "confidence_level": conf_level,
                            "text_length": len(user_input),
                            "word_count": len(user_input.split()),
                            "model": "distilbert-base-uncased-finetuned-imdb"
                        })
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("### 💡 Interpretation")
                    if confidence > 0.9:
                        st.success(f"The model is **very confident** this review is **{label.lower()}**.")
                    elif confidence > 0.7:
                        st.info(f"The model thinks this review is **{label.lower()}** with reasonable confidence.")
                    else:
                        st.warning(f"The model suggests this is **{label.lower()}**, but confidence is relatively low. The sentiment might be mixed or ambiguous.")
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
