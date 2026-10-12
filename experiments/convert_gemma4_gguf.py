"""Run llama.cpp conversion using the Gemma 4 checkpoint's original config.

Recent Transformers migrates global_head_dim to per_layer_config when loading
AutoConfig. Older llama.cpp conversion requires the original global_head_dim.
This adapter preserves config.json in memory; it changes no source files or
weights. Remaining arguments go directly to convert_hf_to_gguf.py.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys


def original_config(directory, is_mistral_format=False):
    if is_mistral_format:
        raise ValueError("This adapter only supports Hugging Face Gemma 4")
    config = json.loads((Path(directory) / "config.json").read_text())
    if config.get("model_type") != "gemma4":
        raise ValueError("Expected a multimodal Gemma 4 checkpoint")
    if not config.get("text_config", {}).get("global_head_dim"):
        raise ValueError("Expected original Gemma 4 global_head_dim configuration")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--llama-source", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    source = args.llama_source.resolve()
    sys.path[:0] = [str(source), str(source / "gguf-py")]
    from conversion.base import ModelBase
    ModelBase.load_hparams = staticmethod(original_config)
    sys.argv = [str(source / "convert_hf_to_gguf.py"), *remaining]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
