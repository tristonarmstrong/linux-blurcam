"""Model download and management."""

import os
from pathlib import Path
import urllib.request


MODEL_REGISTRY = {
    "mediapipe": {
        "source": "huggingface",
        "repo_id": "onnx-community/mediapipe_selfie_segmentation",
        "filename": "onnx/model.onnx",
        "cache_name": "model.onnx",
    },
    "sinet": {
        "source": "url",
        "url": "https://github.com/anilsathyan7/Portrait-Segmentation/raw/master/SINet/SINet.onnx",
        "cache_name": "SINet.onnx",
    },
}


def get_cache_dir() -> Path:
    """Get the cache directory for model files (XDG compliant)."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        cache_dir = Path(xdg_cache) / "blurcam"
    else:
        cache_dir = Path.home() / ".cache" / "blurcam"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_model_path(model_name: str = "mediapipe", force_download: bool = False) -> str:
    """
    Download model if not cached.
    Returns the local path to the model file.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    meta = MODEL_REGISTRY[model_name]
    cache_dir = get_cache_dir()
    model_file = cache_dir / meta["cache_name"]

    if model_file.exists() and not force_download:
        return str(model_file)

    if meta["source"] == "huggingface":
        from huggingface_hub import hf_hub_download

        print(f"Downloading {model_name} model from HuggingFace Hub...")
        print(f"This only happens once. Model will be cached at: {cache_dir}")
        print()

        downloaded_path = hf_hub_download(
            repo_id=meta["repo_id"],
            filename=meta["filename"],
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
        )
        return downloaded_path
    else:
        print(f"Downloading {model_name} model from {meta['url']}...")
        print(f"This only happens once. Model will be cached at: {cache_dir}")
        print()

        urllib.request.urlretrieve(meta["url"], model_file)
        return str(model_file)
