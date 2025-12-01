"""
Streamlit Deployment App for Sentiment Analysis on Hugging Face Spaces
========================================================================

A beautiful web interface for the fine-tuned sentiment analysis model.

Features:
- Clean, modern UI with Streamlit
- Real-time sentiment predictions
- Confidence scores with visual indicators
- Example inputs for quick testing
- Emoji indicators for sentiment
- Optimized for Hugging Face Spaces deployment
"""

import streamlit as st
from transformers import pipeline
import torch
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🎬 Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    """Load the sentiment analysis model (cached for performance)"""
    try:
        # Try to load from Hugging Face Hub first
        model_id = "deshnaashok/sentiment-distilbert-imdb-modern"
        
        # If model is not on HF Hub, use local path (for testing)
        # You'll need to push your model to HF Hub for the Space to work
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
        return sentiment_pipeline, model_id
    except Exception as e:
        # Fallback to a base model if custom model not available
        st.warning(f"⚠️ Custom model not found. Using fallback model. Error: {str(e)}")
        try:
            fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=fallback_model,
                device=0 if torch.cuda.is_available() else -1
            )
            return sentiment_pipeline, fallback_model
        except Exception as fallback_error:
            st.error(f"❌ Error loading fallback model: {str(fallback_error)}")
            return None, None

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Fine-tuned DistilBERT")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app analyzes the sentiment of movie reviews using a fine-tuned DistilBERT model.
        
        **Features:**
        - 🎯 High accuracy sentiment detection
        - 🚀 Fast inference
        - 💡 Confidence scores
        - 🌟 Modern language understanding
        
        **Model:**
        - Base: DistilBERT
        - Training: IMDB dataset
        - Enhanced with modern slang
        """)
        
        st.markdown("---")
        st.markdown("**Tips for best results:**")
        st.markdown("- Write clear, expressive reviews")
        st.markdown("- Include specific details")
        st.markdown("- Express your opinion clearly")
        
        st.markdown("---")
        st.markdown("Made with ❤️ using [Hugging Face](https://huggingface.co)")
    
    # Load model
    with st.spinner("🔄 Loading model..."):
        sentiment_pipeline, model_path = load_model()
    
    if sentiment_pipeline is None:
        st.error("Failed to load model. Please refresh the page or contact support.")
        st.stop()
    
    st.success(f"✅ Model loaded: `{model_path}`")
    
    # Main input area
    st.markdown("### 📝 Enter Your Review")
    
    # Example reviews
    examples = {
        "Select an example...": "",
        "🎭 Positive: Masterpiece": "This movie was an absolute masterpiece! The cinematography was stunning, the acting was superb, and the story kept me on the edge of my seat. Definitely the best film I've seen this year!",
        "🎭 Negative: Disappointing": "A complete waste of time and money. The plot was confusing, the acting was wooden, and I found myself checking my watch every few minutes. Would not recommend to anyone.",
        "🎭 Modern Slang: Fire": "OMG this film was straight fire 🔥! The vibes were immaculate and the plot twist had me shook. It's giving Oscar-worthy performance fr fr. No cap, everyone needs to watch this ASAP!",
        "🎭 Mixed Sentiment": "Great acting from the lead actors, but the script let them down. The visuals were stunning, especially the action sequences, but the story felt rushed and underdeveloped. A missed opportunity.",
        "🎭 Lukewarm": "It was okay, I guess. Not the best movie ever made, but certainly not the worst. Good for a lazy Sunday afternoon when you have nothing better to do.",
        "🎭 Classic Review": "A timeless classic that deserves all the praise it receives. The direction is masterful, every scene is crafted with care, and the performances are unforgettable. This is cinema at its finest."
    }
    
    selected_example = st.selectbox("💡 Choose an example or write your own:", list(examples.keys()))
    
    # Text input
    default_text = examples[selected_example] if selected_example != "Select an example..." else ""
    user_input = st.text_area(
        "Type or paste your movie review here:",
        value=default_text,
        height=150,
        placeholder="e.g., This movie was amazing! The acting was superb and the story kept me engaged throughout...",
        help="Enter your movie review and click 'Analyze Sentiment' to see the results."
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
    
    # Perform analysis
    if analyze_button and user_input.strip():
        with st.spinner("🤔 Analyzing sentiment..."):
            try:
                # Get prediction (limit to 512 tokens for efficiency)
                result = sentiment_pipeline(user_input[:512])[0]
                label = result['label']
                confidence = result['score']
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Sentiment with emoji
                if label.upper() in ["POSITIVE", "LABEL_1"]:
                    sentiment_emoji = "😊"
                    sentiment_color = "green"
                    sentiment_label = "POSITIVE"
                else:
                    sentiment_emoji = "😞"
                    sentiment_color = "red"
                    sentiment_label = "NEGATIVE"
                
                # Confidence level indicator
                if confidence > 0.85:
                    conf_indicator = "🟢"
                    conf_level = "Very High"
                elif confidence > 0.70:
                    conf_indicator = "🟡"
                    conf_level = "High"
                elif confidence > 0.55:
                    conf_indicator = "🟠"
                    conf_level = "Medium"
                else:
                    conf_indicator = "🔴"
                    conf_level = "Low"
                
                # Display in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f'<div class="sentiment-box" style="background-color: #e8f5e9;">', unsafe_allow_html=True)
                    st.markdown(f"### {sentiment_emoji} Sentiment")
                    st.markdown(f"# :{sentiment_color}[**{sentiment_label}**]")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'<div class="sentiment-box" style="background-color: #e3f2fd;">', unsafe_allow_html=True)
                    st.markdown(f"### {conf_indicator} Confidence")
                    st.markdown(f"# **{confidence:.1%}**")
                    st.markdown(f"*{conf_level} confidence*")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Progress bar for confidence
                st.progress(confidence)
                
                # Interpretation
                st.markdown("---")
                st.markdown("### 💡 Interpretation")
                
                if confidence > 0.90:
                    st.success(f"🎯 The model is **very confident** this review is **{sentiment_label.lower()}**. The sentiment is clear and strong.")
                elif confidence > 0.75:
                    st.info(f"✅ The model thinks this review is **{sentiment_label.lower()}** with good confidence. The sentiment is reasonably clear.")
                elif confidence > 0.60:
                    st.warning(f"⚠️ The model suggests this is **{sentiment_label.lower()}**, but confidence is moderate. The review might contain mixed sentiments.")
                else:
                    st.warning(f"❓ The model leans toward **{sentiment_label.lower()}**, but confidence is low. This review likely has ambiguous or mixed sentiment.")
                
                # Additional info
                with st.expander("📈 Detailed Metrics"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Text Length", f"{len(user_input)} chars")
                    with col_b:
                        st.metric("Word Count", f"{len(user_input.split())} words")
                    with col_c:
                        st.metric("Tokens Used", min(512, len(user_input.split())))
                    
                    st.json({
                        "sentiment": sentiment_label,
                        "confidence_score": round(confidence, 4),
                        "confidence_level": conf_level,
                        "original_label": label,
                        "model": model_path
                    })
                
                # Call to action
                st.markdown("---")
                st.markdown("### 🎬 Try Another Review!")
                st.markdown("Select a different example above or write your own review to analyze.")
            
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.error("Please try again or contact support if the issue persists.")
    
    elif analyze_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>Built with Streamlit • Powered by Hugging Face Transformers</p>
        <p>© 2025 Movie Sentiment Analysis</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
