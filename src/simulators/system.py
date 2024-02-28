from abc import abstractmethod, ABC
import numpy as np
from src.simulators.types import SimTime, SysDim, SigParam
from typing import Optional, Callable, Union, Any


class System(ABC):
    @abstractmethod
    def diff_eq(self):
        pass

    @abstractmethod
    def generate_ic(self, num_samples: int):
        pass

    @abstractmethod
    def get_output(self, states: np.ndarray):
        pass


class InputSignal(ABC):
    @abstractmethod
    def generate_ic(self, num_samples: int):
        pass

    @abstractmethod
    def generate_signal(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def _check_data(self, data: dict) -> None:
        pass


class SinSignal(InputSignal):
    def __init__(self, sampler: Callable, sig_param: SigParam, num_samples: int = 1):
        self._check_data(sig_param)
        self.signal_data = sig_param.signal_data
        self.sampler = sampler
        self._ic = sampler(num_samples)

    def generate_signal(self, t: Union[float, np.ndarray], ic: float = 0) -> Union[float, np.ndarray]:
        input_signal = 0
        for values in zip(*self.signal_data.values()):
            amp, freq, phase = values
            input_signal += amp * np.sin(2 * np.pi * freq * t + ic * np.pi)
        return np.array(input_signal)

    def generate_trajs(self, sim_time: SimTime) -> np.ndarray:
        """ Generate the input signal """
        t = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
        trajectory = []
        for ic in self._ic:
            trajectory.append(self.generate_signal(t, ic))  # phase is assumed to be the random variable
        return np.array(trajectory)

    def _check_data(self, data) -> None:
        assert data.signal_type == "harmonics", "This signal generator only supports harmonics"
        assert all(key in data.signal_data for key in
                   ['amp', 'freq', 'phase']), "Required keys ('amp', 'freq', 'phase') are missing in signal_data"
        values = list(data.signal_data.values())
        assert all(len(value) == len(values[0]) for value in values), "Values in signal_data should have equal lengths"

    def generate_ic(self, num_samples: int) -> None:
        self._ic = self.sampler(num_samples)

    @property
    def ic(self) -> np.ndarray:
        return self._ic


class Duffing(System):

    def __init__(self, sampler, system_param, num_samples=1, noise: Callable = None) -> None:
        # calculate the z_dim
        self.sampler = sampler
        self.noise = noise if noise else lambda x, t: np.zeros_like(x)
        self._ic = sampler(num_samples)
        self.system_param = system_param

    def generate_ic(self, num_samples: int) -> None:
        self._ic = self.sampler(num_samples)

    def diff_eq(self, t: float, x: list[float], inp: Optional[Union[float, Callable]] = 0) -> np.ndarray:
        """ System function """

        if callable(inp): inp = inp(t)
        x1_dot = x[1] ** 3
        x2_dot = - x[0] + inp

        return np.array([x1_dot, x2_dot]) + self.noise(x, t)

    def get_output(self, states: np.ndarray) -> np.ndarray:
        """ Returns the output of the system """
        return np.array(self.system_param.C) * states

    @property
    def ic(self) -> np.ndarray:
        return self._ic

    @property
    def sys_dim(self) -> SysDim:
        """
        Calculated as mentioned in https://arxiv.org/pdf/2210.01476.pdf
        nz = ny(2nx + 1) # in case of autonomous system
        """
        x_dim, y_dim = 2, 1
        z_dim = y_dim * (2 * x_dim + 1)
        return SysDim(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)


class Observer:
    def __init__(self, A: list, B: list, z_dim: int, e: float, z_max: int, sampler, num_samples=1) -> None:
        self.A = np.array(A).reshape(z_dim, z_dim)
        self.B = np.array(B)
        self.z_dim = z_dim
        self._ic = sampler(num_samples)
        self.sampler = sampler
        self.e = e
        self.z_max = z_max

    def generate_ic(self, num_samples: int) -> None:
        self._ic = self.sampler(num_samples)

    def diff_eq(self, t: float, z: list[float], y: list[float]) -> tuple[float]:
        z_dot = np.matmul(self.A, z) + self.B * y
        return z_dot

    # function related to the observer
    def calc_pret0(self):
        w, v = np.linalg.eig(self.A)
        min_ev = np.min(np.abs(np.real(w)))
        kappa = np.linalg.cond(v)
        s = np.sqrt(
            self.z_max * self.A.shape[0])  # check this: https://en.wikipedia.org/wiki/Norm_(mathematics)#Properties
        t = 1 / min_ev * np.log(self.e / (kappa * s))
        return t

    @property
    def ic(self):
        return self._ic
