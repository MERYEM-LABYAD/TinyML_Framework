import torch.nn as nn


class Teacher:
    """
    Teacher = M0.

    Toujours frozen et en eval().
    """

    def __init__(self, model: nn.Module):
        self.model = model

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.eval()

    def parameters(self):
        return self.model.parameters()