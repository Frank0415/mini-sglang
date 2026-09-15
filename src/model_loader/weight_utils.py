import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open

def iter_weights(model_dir: str|Path) -> Iterator[tuple[str, torch.Tensor]]:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"

    if index_path.is_file():
        with index_path.open(encoding="utf-8") as file:
            index = json.load(file)

        filenames = sorted(set(index["weight_map"].values()))
    else:
        filenames = ["model.safetensors"]

    for filename in filenames:
        weight_path = model_dir / filename

        with safe_open(
            str(weight_path),
            framework="pt",
            device="cpu",
        ) as file:
            for name in file.keys():  # noqa: SIM118
                yield name, file.get_tensor(name)
    