"""
Upload fine-tuned sentiment analysis model to Hugging Face Hub
"""
from huggingface_hub import HfApi, create_repo
import os

# Initialize API
api = HfApi()

# Configuration
username = "deshnaashok"  # Your HF username
model_name = "sentiment-distilbert-imdb-modern"
model_path = "models/sentiment-distilbert-imdb-modern"

print("🚀 Uploading Model to Hugging Face Hub")
print("=" * 50)
print(f"Username: {username}")
print(f"Model: {model_name}")
print(f"Local path: {model_path}")
print()

# Step 1: Create model repository
print("📦 Step 1: Creating model repository...")
try:
    repo_url = create_repo(
        repo_id=f"{username}/{model_name}",
        repo_type="model",
        exist_ok=True,
        private=False
    )
    print(f"✅ Repository created/exists: {repo_url}")
except Exception as e:
    print(f"❌ Error creating repository: {e}")
    exit(1)

# Step 2: Upload model files
print("\n📤 Step 2: Uploading model files...")
try:
    api.upload_folder(
        folder_path=model_path,
        repo_id=f"{username}/{model_name}",
        repo_type="model"
    )
    print("✅ Model files uploaded successfully!")
except Exception as e:
    print(f"❌ Error uploading model: {e}")
    exit(1)

print("\n" + "=" * 50)
print("🎉 Model Upload Complete!")
print(f"🔗 View your model: https://huggingface.co/{username}/{model_name}")
print("=" * 50)
