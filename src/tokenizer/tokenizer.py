from pathlib import Path

from transformers import AutoTokenizer


def load_tokenizer(model_dir: str | Path):
    return AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
