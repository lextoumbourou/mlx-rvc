#!/usr/bin/env python3
"""Upload RVC-MLX weights to HuggingFace Hub."""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID = "lexandstuff/rvc-mlx-weights"
WEIGHTS_DIR = Path("weights")


def main():
    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/accessing repo: {REPO_ID}")
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Repo may already exist: {e}")

    # Upload all files
    files_to_upload = [
        ("README.md", "README.md"),
        ("v2/config.json", "v2/config.json"),
        ("v2/f0G32k.safetensors", "v2/f0G32k.safetensors"),
        ("v2/f0G40k.safetensors", "v2/f0G40k.safetensors"),
        ("v2/f0G48k.safetensors", "v2/f0G48k.safetensors"),
    ]

    for local_path, repo_path in files_to_upload:
        full_path = WEIGHTS_DIR / local_path
        if not full_path.exists():
            print(f"Skipping {local_path} (not found)")
            continue

        print(f"Uploading {local_path} -> {repo_path}")
        api.upload_file(
            path_or_fileobj=str(full_path),
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
        )

    print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
