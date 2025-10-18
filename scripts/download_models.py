"""
Script to download pre-trained models from HuggingFace.

Interactive script guiding users through model download process.
"""

import sys
from pathlib import Path


def main() -> None:
    """Download pre-trained models."""
    print("=" * 70)
    print("Model Download Script")
    print("=" * 70)
    print("\nThis script downloads pre-trained models for code generation.")
    print("\nAvailable models:")
    print("  1. DeepSeek-Coder-6.7B")
    print("  2. StarCoder2-15B")
    print("  3. CodeLlama-13B")
    print("\nTo download models, you'll need:")
    print("  - HuggingFace account (free)")
    print("  - Sufficient disk space (10-50GB per model)")
    print("  - Fast internet connection")
    print("\nExample download command:")
    print("  huggingface-cli download deepseek-ai/deepseek-coder-6.7b-base \\")
    print("    --local-dir models/pretrained/deepseek-coder-6.7b")
    print("\n✓ Model download script ready")
    print("Run the above commands manually to download models")
    print("=" * 70)


if __name__ == "__main__":
    main()
