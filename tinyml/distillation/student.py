import torch.nn as nn


class Student:
    """
    Student = M2.

    Les paramètres restent entraînables.
    """

    def __init__(self, model: nn.Module):
        self.model = model

        for parameter in self.model.parameters():
            parameter.requires_grad = True

        self.model.train()

    def parameters(self):
        return self.model.parameters()