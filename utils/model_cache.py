"""
Simple model caching system for PSTXM.
Automatically caches the first loaded model and reuses it for subsequent runs.
When switching to a different model, clears the old cache and caches the new one.
"""

import os
import torch
import pickle
import time
from pathlib import Path
from transformers import GPT2Model, AutoModel


class ModelCache:
    """Global model cache that persists across runs."""

    _instance = None
    _cache_dir = os.path.join(Path.home(), ".cache", "pstxm_model_memory")
    _cache_file = os.path.join(_cache_dir, "cached_model.pkl")
    _info_file = os.path.join(_cache_dir, "cache_info.pkl")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        os.makedirs(self._cache_dir, exist_ok=True)
        self.cached_model = None
        self.cache_info = self._load_cache_info()

    def _load_cache_info(self):
        """Load cache metadata."""
        if os.path.exists(self._info_file):
            try:
                with open(self._info_file, "rb") as f:
                    return pickle.load(f)
            except:
                pass
        return {}

    def _save_cache_info(self):
        """Save cache metadata."""
        with open(self._info_file, "wb") as f:
            pickle.dump(self.cache_info, f)

    def _get_model_key(self, model_type, model_path):
        """Generate unique key for a model."""
        return f"{model_type}:{model_path}"

    def get_or_load_model(self, model_type, model_path, model_config):
        """
        Get model from cache or load it.
        If a different model is requested, clear old cache and load new one.

        Returns:
            tuple: (model, from_cache) where from_cache indicates if model was cached
        """
        model_key = self._get_model_key(model_type, model_path)
        current_cached_key = self.cache_info.get("model_key")

        # Check if we have the right model cached
        if current_cached_key == model_key and self.cached_model is not None:
            print(f"Using cached {model_type} model (already in memory)")
            return self.cached_model, True

        # If we have a different model cached, clear it
        if current_cached_key and current_cached_key != model_key:
            print(f"Clearing old cached model: {current_cached_key}")
            self.clear_cache()

        # Try to load from disk cache
        if (
            os.path.exists(self._cache_file)
            and self.cache_info.get("model_key") == model_key
        ):
            try:
                print(f"Loading {model_type} from disk cache...")
                start_time = time.time()
                with open(self._cache_file, "rb") as f:
                    self.cached_model = pickle.load(f)
                load_time = time.time() - start_time
                print(f"✓ Loaded from disk cache in {load_time:.2f}s")
                return self.cached_model, True
            except Exception as e:
                print(f"Failed to load from disk cache: {e}")
                self.clear_cache()

        # Load new model
        print(f"Loading {model_type} model for the first time...")
        start_time = time.time()

        # Prepare loading arguments
        common_args = {
            "output_attentions": True,
            "output_hidden_states": True,
            "low_cpu_mem_usage": True,
        }

        # Add local files only if it's a local model
        is_local = os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")
        )
        if is_local:
            common_args["local_files_only"] = True

        # Load model based on type
        if model_type == "gpt2":
            model = GPT2Model.from_pretrained(model_path, **common_args)
        elif model_type == "qwen3-0.6b":
            common_args.update(
                {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16,
                    "attn_implementation": "eager",
                }
            )
            model = AutoModel.from_pretrained(model_path, **common_args)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")

        # Cache the model
        self.cached_model = model
        self.cache_info = {
            "model_key": model_key,
            "model_type": model_type,
            "model_path": model_path,
            "cached_at": time.time(),
        }
        self._save_cache_info()

        # Save to disk cache in background
        print("Saving model to disk cache for future runs...")
        try:
            with open(self._cache_file, "wb") as f:
                pickle.dump(model, f)
            print("✓ Model cached to disk")
        except Exception as e:
            print(f"Warning: Failed to cache model to disk: {e}")

        return model, False

    def clear_cache(self):
        """Clear all cached models."""
        self.cached_model = None
        self.cache_info = {}

        # Remove cache files
        for file in [self._cache_file, self._info_file]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass

        # Clear GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Model cache cleared")


# Global cache instance
_model_cache = ModelCache()


def get_cached_model(model_type, model_path, model_config=None):
    """
    Simple interface to get a cached model.

    Args:
        model_type: Type of model (gpt2, qwen3-0.6b, etc.)
        model_path: Path to the model
        model_config: Optional model configuration

    Returns:
        tuple: (model, from_cache)
    """
    return _model_cache.get_or_load_model(model_type, model_path, model_config)


def clear_model_cache():
    """Clear the model cache."""
    _model_cache.clear_cache()
