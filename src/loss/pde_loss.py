import torch.nn as nn, torch


class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.std = std

    def normalize(self, tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor) -> torch.Tensor:
        return (tensor * self.std) + self.mean


class PDELoss(torch.nn.Module):
    def __init__(self, observer: Observer):
        super().__init__()
        self.observer: Observer = observer

    def forward(self, model: torch.nn.Module, x_label: torch.Tensor, x_dot_label: torch.Tensor, y_label: torch.Tensor,
                z_pred: torch.Tensor) -> torch.Tensor:
        ...
        return loss_pde
