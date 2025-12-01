"""
Gradio Deployment App for Sentiment Analysis on Hugging Face Spaces
=====================================================================

A beautiful web interface for the fine-tuned sentiment analysis model.

Features:
- Clean, modern UI with Gradio
- Real-time sentiment predictions
- Confidence scores with visual indicators
- Example inputs for quick testing
- Emoji indicators for sentiment
- Optimized for Hugging Face Spaces deployment
"""

import gradio as gr
from transformers import pipeline
import torch

# ============================================================================
# LOAD MODEL
# ============================================================================
def load_model():
    """Load the sentiment analysis model"""
    try:
        # Load from Hugging Face Hub with authentication token if available
        model_id = "deshnaashok/sentiment-distilbert-imdb-modern"
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            use_auth_token=True  # Use HF Space's token
        )
        print(f"✅ Successfully loaded custom model: {model_id}")
        return sentiment_pipeline, model_id
    except Exception as e:
        # Fallback to a base model if custom model not available
        print(f"⚠️ Custom model not found. Using fallback model. Error: {str(e)}")
        try:
            fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=fallback_model,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ Loaded fallback model: {fallback_model}")
            return sentiment_pipeline, fallback_model
        except Exception as fallback_error:
            print(f"❌ Error loading fallback model: {str(fallback_error)}")
            return None, None

# Load model at startup
print("�� Loading model...")
sentiment_pipeline, model_id = load_model()
print(f"✅ Model loaded: {model_id}")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def analyze_sentiment(text):
    """Analyze the sentiment of the input text"""
    if not text or not text.strip():
        return "⚠️ Please enter some text to analyze.", "", ""
    
    try:
        # Get prediction (limit to 512 tokens)
        result = sentiment_pipeline(text[:512])[0]
        label = result['label']
        confidence = result['score']
        
        # Determine sentiment
        if label.upper() in ["POSITIVE", "LABEL_1"]:
            sentiment_emoji = "😊"
            sentiment_label = "POSITIVE"
            sentiment_color = "#4CAF50"
        else:
            sentiment_emoji = "😞"
            sentiment_label = "NEGATIVE"
            sentiment_color = "#F44336"
        
        # Confidence level
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
        
        # Format output
        sentiment_output = f"""
        <div style="text-align: center; padding: 20px; background-color: {sentiment_color}20; border-radius: 10px; border: 2px solid {sentiment_color};">
            <h2 style="color: {sentiment_color};">{sentiment_emoji} {sentiment_label}</h2>
        </div>
        """
        
        confidence_output = f"""
        <div style="text-align: center; padding: 20px; background-color: #2196F320; border-radius: 10px; border: 2px solid #2196F3;">
            <h2>{conf_indicator} Confidence</h2>
            <h1 style="color: #2196F3;">{confidence:.1%}</h1>
            <p style="font-size: 18px;">{conf_level} confidence</p>
        </div>
        """
        
        # Interpretation
        if confidence > 0.90:
            interpretation = f"🎯 The model is **very confident** this review is **{sentiment_label.lower()}**. The sentiment is clear and strong."
        elif confidence > 0.75:
            interpretation = f"✅ The model thinks this review is **{sentiment_label.lower()}** with good confidence. The sentiment is reasonably clear."
        elif confidence > 0.60:
            interpretation = f"⚠️ The model suggests this is **{sentiment_label.lower()}**, but confidence is moderate. The review might contain mixed sentiments."
        else:
            interpretation = f"❓ The model leans toward **{sentiment_label.lower()}**, but confidence is low. This review likely has ambiguous or mixed sentiment."
        
        return sentiment_output, confidence_output, interpretation
    
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}", "", ""

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Example reviews
examples = [
    ["This movie was an absolute masterpiece! The cinematography was stunning, the acting was superb, and the story kept me on the edge of my seat. Definitely the best film I've seen this year!"],
    ["A complete waste of time and money. The plot was confusing, the acting was wooden, and I found myself checking my watch every few minutes. Would not recommend to anyone."],
    ["OMG this film was straight fire 🔥! The vibes were immaculate and the plot twist had me shook. It's giving Oscar-worthy performance fr fr. No cap, everyone needs to watch this ASAP!"],
    ["Great acting from the lead actors, but the script let them down. The visuals were stunning, especially the action sequences, but the story felt rushed and underdeveloped. A missed opportunity."],
    ["It was okay, I guess. Not the best movie ever made, but certainly not the worst. Good for a lazy Sunday afternoon when you have nothing better to do."],
    ["A timeless classic that deserves all the praise it receives. The direction is masterful, every scene is crafted with care, and the performances are unforgettable. This is cinema at its finest."]
]

# Create Gradio interface
with gr.Blocks(title="🎬 Movie Sentiment Analysis") as demo:
    gr.Markdown(
        """
        # 🎬 Movie Review Sentiment Analysis
        ### Powered by Fine-tuned DistilBERT
        
        Analyze the sentiment of movie reviews using a fine-tuned DistilBERT model trained on the IMDB dataset.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Enter Your Review")
            text_input = gr.Textbox(
                label="Movie Review",
                placeholder="e.g., This movie was amazing! The acting was superb and the story kept me engaged throughout...",
                lines=6
            )
            
            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
            
            gr.Markdown("### 💡 Or Try These Examples:")
            gr.Examples(
                examples=examples,
                inputs=text_input,
                label="Example Reviews"
            )
    
    gr.Markdown("---")
    gr.Markdown("## 📊 Analysis Results")
    
    with gr.Row():
        with gr.Column():
            sentiment_output = gr.HTML(label="Sentiment")
        with gr.Column():
            confidence_output = gr.HTML(label="Confidence")
    
    interpretation_output = gr.Markdown(label="Interpretation")
    
    gr.Markdown("---")
    
    with gr.Accordion("ℹ️ About This Model", open=False):
        gr.Markdown(
            f"""
            **Model:** `{model_id}`
            
            **Features:**
            - 🎯 High accuracy sentiment detection (~93% on IMDB test set)
            - 🚀 Fast inference with DistilBERT
            - 💡 Confidence scores for transparency
            - 🌟 Enhanced with modern language understanding
            
            **Training:**
            - Base: DistilBERT (distilbert-base-uncased)
            - Dataset: IMDB Movie Reviews (50,000 reviews)
            - Fine-tuned with modern slang and expressions
            
            **Tips for best results:**
            - Write clear, expressive reviews
            - Include specific details about the movie
            - Express your opinion clearly
            """
        )
    
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #666;">
            <p>Built with Gradio • Powered by Hugging Face Transformers</p>
            <p>© 2025 Movie Sentiment Analysis</p>
        </div>
        """
    )
    
    # Connect the button to the function
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, interpretation_output]
    )

# ============================================================================
# LAUNCH APP
# ============================================================================
if __name__ == "__main__":
    demo.launch()
