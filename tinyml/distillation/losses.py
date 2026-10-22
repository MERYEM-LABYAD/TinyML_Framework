from typing import Dict

import torch
import torch.nn.functional as F


def task_loss_from_output(output):
    """
    Detectron2 retourne normalement un dict de losses en training.
    Pour un modèle générique, un tensor peut également être retourné.
    """

    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        losses = [
            value
            for value in output.values()
            if torch.is_tensor(value)
        ]

        if not losses:
            raise ValueError(
                "No tensor loss found in model output."
            )

        return sum(losses)

    raise TypeError(
        f"Unsupported task output type: {type(output)}"
    )


def feature_distillation_loss(
    student_features: Dict[str, torch.Tensor],
    teacher_features: Dict[str, torch.Tensor],
):
    losses = []

    for name in teacher_features:
        if name not in student_features:
            raise KeyError(
                f"Student feature missing: {name}"
            )

        student = student_features[name]
        teacher = teacher_features[name].detach()

        if student.shape != teacher.shape:
            raise ValueError(
                f"Feature shape mismatch for {name}: "
                f"student={tuple(student.shape)}, "
                f"teacher={tuple(teacher.shape)}"
            )

        losses.append(
            F.mse_loss(student, teacher)
        )

    if not losses:
        raise ValueError(
            "No feature available for distillation."
        )

    return sum(losses) / len(losses)


def combined_distillation_loss(
    task_loss,
    distillation_loss,
    alpha: float = 0.5,
):
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    return (
        alpha * task_loss
        + (1.0 - alpha) * distillation_loss
    )