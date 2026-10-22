import os
import tempfile
import torch


def model_size_bytes(model) -> int:
    with tempfile.NamedTemporaryFile(
        suffix=".pth",
        delete=False,
    ) as temporary:
        path = temporary.name

    try:
        torch.save(
            model.state_dict(),
            path,
        )

        return os.path.getsize(path)

    finally:
        if os.path.exists(path):
            os.remove(path)


def model_size_mb(model) -> float:
    return model_size_bytes(model) / (
        1024 * 1024
    )


def size_reduction(
    baseline_bytes,
    current_bytes,
):
    if baseline_bytes <= 0:
        raise ValueError(
            "baseline_bytes must be > 0."
        )

    return (
        1.0
        - current_bytes / baseline_bytes
    ) * 100.0