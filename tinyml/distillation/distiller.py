from typing import Dict

import torch
import torch.nn as nn

from .losses import (
    task_loss_from_output,
    feature_distillation_loss,
    combined_distillation_loss,
)


class FeatureCollector:
    """
    Collecte les sorties de modules désignés par leur nom.
    """

    def __init__(
        self,
        model: nn.Module,
        module_names,
    ):
        self.model = model
        self.module_names = list(module_names)
        self.features = {}
        self.handles = []

        modules = dict(model.named_modules())

        for name in self.module_names:
            if name not in modules:
                raise KeyError(
                    f"Feature module not found: {name}"
                )

            handle = modules[name].register_forward_hook(
                self._make_hook(name)
            )

            self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                self.features[name] = output

        return hook

    def clear(self):
        self.features.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()


class Distiller:
    """
    Teacher = M0
    Student = M2

    Le teacher ne reçoit aucun gradient.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        feature_modules,
        alpha: float = 0.5,
    ):
        self.teacher = teacher
        self.student = student
        self.alpha = alpha

        self.teacher.eval()

        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.student.train()

        self.teacher_features = FeatureCollector(
            self.teacher,
            feature_modules,
        )

        self.student_features = FeatureCollector(
            self.student,
            feature_modules,
        )

    def step(self, batch):
        self.teacher_features.clear()
        self.student_features.clear()

        with torch.no_grad():
            self.teacher(batch)

        student_output = self.student(batch)

        task_loss = task_loss_from_output(
            student_output
        )

        distill_loss = feature_distillation_loss(
            self.student_features.features,
            self.teacher_features.features,
        )

        total_loss = combined_distillation_loss(
            task_loss=task_loss,
            distillation_loss=distill_loss,
            alpha=self.alpha,
        )

        return {
            "task_loss": task_loss,
            "distillation_loss": distill_loss,
            "total_loss": total_loss,
        }

    def close(self):
        self.teacher_features.remove()
        self.student_features.remove()