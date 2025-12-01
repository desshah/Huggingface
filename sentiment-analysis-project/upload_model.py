"""
Upload Gradio app to Hugging Face Spaces
"""
from huggingface_hub import HfApi, create_repo
import os

# Initialize API
api = HfApi()

# Configuration
username = "deshnaashok"  # Your HF username
space_name = "movie-sentiment-analysis"
space_path = "deployment"  # Path to deployment folder (like model_path in upload_model.py)

print("🚀 Uploading Space to Hugging Face")
print("=" * 50)
print(f"Username: {username}")
print(f"Space: {space_name}")
print(f"Local path: {space_path}")
print()

# Step 1: Create Space repository
print("📦 Step 1: Creating Space repository...")
try:
    repo_url = create_repo(
        repo_id=f"{username}/{space_name}",
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False
    )
    print(f"✅ Repository created/exists: {repo_url}")
except Exception as e:
    print(f"❌ Error creating repository: {e}")
    exit(1)

# Step 2: Upload Space files
print("\n📤 Step 2: Uploading Space files...")
try:
    api.upload_folder(
        folder_path=space_path,
        repo_id=f"{username}/{space_name}",
        repo_type="space",
        ignore_patterns=["*.pyc", "__pycache__", ".git", "upload_space.py", "streamlit_app.py", "*.md", "app_streamlit_backup.py"]
    )
    print("✅ Space files uploaded successfully!")
except Exception as e:
    print(f"❌ Error uploading space: {e}")
    exit(1)

print("\n" + "=" * 50)
print("🎉 Space Upload Complete!")
print(f"🔗 View your space: https://huggingface.co/spaces/{username}/{space_name}")
print("=" * 50)