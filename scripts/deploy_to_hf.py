"""
deploy_to_hf.py  —  Upload trained model files + app code to HuggingFace.
Called by the GitHub Actions deploy job.
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ── read env vars ──────────────────────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "").strip()
HF_USERNAME = os.environ.get("HF_USERNAME", "").strip()

if not HF_TOKEN or not HF_USERNAME:
    print("ERROR: HF_TOKEN and HF_USERNAME secrets must be set in GitHub.")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)

# ── 1. Push .pkl models to HF Hub Model Registry ──────────────────────────
model_repo = f"{HF_USERNAME}/sales-sentiment-model"
create_repo(model_repo, token=HF_TOKEN, repo_type="model", exist_ok=True)
print(f"Model registry: https://huggingface.co/{model_repo}")

model_files = [
    "sentiment_model_5000.pkl",
    "tfidf_vectorizer_5000.pkl",
    "label_encoder_5000.pkl",
    "training_metadata.json",
]

for fname in model_files:
    fpath = Path("app") / "models" / fname
    if fpath.exists():
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=model_repo,
            repo_type="model",
        )
        print(f"  Uploaded model file: {fname}")
    else:
        print(f"  WARNING — not found, skipping: {fpath}")

print("Model registry updated!\n")

# ── 2. Deploy app to HF Spaces ────────────────────────────────────────────
space_repo = f"{HF_USERNAME}/sales-sentiment-api"
create_repo(
    space_repo,
    token=HF_TOKEN,
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)
print(f"Space: https://huggingface.co/spaces/{space_repo}")

# Upload root files
for root_file in ["Dockerfile", "requirements.txt"]:
    if Path(root_file).exists():
        api.upload_file(
            path_or_fileobj=root_file,
            path_in_repo=root_file,
            repo_id=space_repo,
            repo_type="space",
        )
        print(f"  Uploaded: {root_file}")

# Upload entire app folder (code + models + templates + static)
api.upload_folder(
    folder_path="app",
    path_in_repo="app",
    repo_id=space_repo,
    repo_type="space",
)
print("  Uploaded: app/ folder (code + models)")

print("\n" + "=" * 55)
print("DEPLOYMENT COMPLETE!")
print(f"  Model registry : https://huggingface.co/{model_repo}")
print(f"  Live Space     : https://huggingface.co/spaces/{space_repo}")
print("=" * 55)
