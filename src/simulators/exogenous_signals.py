from abc import abstractmethod, ABC
import numpy as np
from src.simulators.types import SimTime, SigParam
from typing import Callable, Union


class InputSignal(ABC):
    def __init__(self, sampler: Callable, sig_param: SigParam, num_samples: int = 1):
        self._check_data(sig_param)
        self.signal_data = sig_param.signal_data
        self.sampler = sampler
        self._ic = sampler(num_samples)

    def generate_ic(self, num_samples: int):
        self._ic = self.sampler(num_samples)

    @abstractmethod
    def generate_trajs(self, sim_time: SimTime) -> np.ndarray:
        pass

    @abstractmethod
    def generate_signal(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def _check_data(self, data: SigParam) -> None:
        pass

    @property
    def ic(self) -> np.ndarray:
        return self._ic


class SinSignal(InputSignal):
    def __init__(self, sampler: Callable, sig_param: SigParam, num_samples: int = 1):
        super().__init__(sampler, sig_param, num_samples)

    def generate_signal(self, t: Union[float, np.ndarray], ic: float = 0) -> Union[float, np.ndarray]:
        input_signal = 0
        for values in zip(*self.signal_data.values()):
            amp, freq, phase = values
            input_signal += amp * np.sin(2 * np.pi * freq * t + ic * np.pi)
        return np.array(input_signal).reshape(t.shape[0], -1)

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
