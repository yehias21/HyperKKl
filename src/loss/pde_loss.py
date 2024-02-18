import torch
import numpy as np
from torch.autograd import function
from typing import cast


class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.std = std

    def normalize(self, tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor) -> torch.Tensor:
        return (tensor * self.std) + self.mean


class Criterion(torch.nn.Module):
    def __init__(self, loss:str, method, pde_loss= None):
        super().__init__()
        self.pde_loss = pde_loss
        self.method = method
        match loss:
            case "mse":
                self.loss_fn = torch.nn.MSELoss()
            case _:
                raise NotImplementedError(f"Loss {loss} not supported")

    def forward(self, data, output) -> torch.Tensor:
        match self.method:
            case "unsupervised":
                loss = self.loss_fn(output['x'], data['x'])
            case "supervised":
                loss = self.loss_fn(output['x'], data['x'])
                loss += self.loss_fn(output['z'], data['z'])
            case "supervised_pinn":
                loss = self.loss_fn(output['x'], data['x'])
                loss += self.loss_fn(output['z'], data['z'])
                loss += self.pde_loss(output['z'], data['x'], data['y'])
            case _:
                raise NotImplementedError(f"Method {self.method} not supported")
        return loss


class PDELoss(torch.nn.Module):
    def __init__(self, diff_eq, forward_mapper, A, B):
        super().__init__()
        self.diff_eq = diff_eq
        self.forward_mapper = forward_mapper
        self.A, self.B = torch.from_numpy(A.astype(dtype=np.float32)), torch.from_numpy(B.astype(dtype=np.float32))

    def _get_jac(self, x):
        dtdx = torch.autograd.functional.jacobian(cast(function, self.forward_mapper), x)
        idx = torch.arange(x.shape[0])
        dtdx = dtdx[idx, :, idx, :]
        return dtdx

    def forward(self,
                z_pred: torch.Tensor, x_label, y_label) -> torch.Tensor:
        """
         mean squared residual : N
        """
        assert x_label.shape == 2
        x_dot = np.array([self.diff_eq(x) for x in x_label])
        dttheta_dx = self._get_jac(x_label)
        dtdx_mul_f = torch.bmm(dttheta_dx, torch.unsqueeze(x_dot, 2))

        m_mul_t = torch.matmul(self.A, torch.unsqueeze(z_pred, 2))
        k_mul_h = torch.matmul(self.B, torch.unsqueeze(y_label, 2))

        pde = dtdx_mul_f - m_mul_t - k_mul_h
        loss_batch = torch.linalg.norm(pde, dim=1)  # fixme: is order 2 or norm is enough
        loss_pde = torch.mean(loss_batch)

        return loss_pde
