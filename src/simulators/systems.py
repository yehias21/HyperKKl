from abc import abstractmethod, ABC
import numpy as np
from src.simulators.types import SysDim, SysParam
from typing import Optional, Callable, Union


class System(ABC):
    """
    Abstract class for the system, this class should be inherited by the system class.
    """

    @abstractmethod
    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None) -> None:
        """
        In our implementation, we chose to follow dependency injection design pattern, where the sampler, and
        the noise classes are initialized outside the system class, their callables (same paradigm of forward functions
        in pytorch) are passed to the system class.
        Therefore, a change in the sampler or the noise class will not affect the system class.
        :param sampler: is Callable, which samples the initial conditions of the system
        :param system_param: is a DataClass, that has the system parameters of the differential equation,\
        the C matrix (gain of the sensors), the observable index (index of the states that are observable)
        :param num_samples: is the number of the initial conditions we want the system to have (number of trajectories)
        :param p_noise, m_noise: are 2 callables, first that adds noise to the system dynamics (process noise),
        and the second that adds noise to the output of the system (measurement/sensor noise)
        The noise functions is designed to be non-stationary (i.e. time-varying, state dependent) for generalization
        """
        self._sampler = sampler
        self.noise = (p_noise if p_noise else lambda x, t: np.zeros_like(x)
                      , m_noise if m_noise else lambda x: np.zeros_like(x))
        self._ic = sampler(num_samples)
        self.system_param = system_param

    def generate_ic(self, num_samples: int) -> None:
        """
           Generate the initial conditions of the system
           :param num_samples: number of initial conditions
           :return: numpy array of initial conditions of the system
       """
        self._ic = self._sampler(num_samples)

    def get_output(self, states: np.ndarray) -> np.ndarray:
        """ Returns the output of the system """
        return (np.array(self.system_param.C) * states + + self.noise[1](states))[..., self.system_param.ObservableIndex]

    @property
    def ic(self) -> np.ndarray:
        return self._ic

    def sampler(self, sampler: Callable) -> None:
        self._sampler = sampler

    @abstractmethod
    def diff_eq(self, t: float, x: list[float], inp: Optional[Union[float, Callable]] = 0) -> np.ndarray:
        """
        Set of differential equation that describes the system dynamics
        :param t: time
        :param x: current states of the system
        :param inp: exogenous input of the system in case it's non-autonomous
        :return: numpy array of the derivatives of the states at time t (+ noise)
        """
        pass

    @abstractmethod
    def sys_dim(self) -> SysDim:
        """
        Returns the system dimensions
        """
        pass


class Duffing(System):

    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None) -> None:
        super().__init__(sampler, system_param, num_samples, p_noise, m_noise)

    def diff_eq(self, t: float, x: list[float], inp: Optional[Union[float, Callable]] = 0) -> np.ndarray:
        if callable(inp): inp = inp(t)
        x1_dot = x[1] ** 3
        x2_dot = - x[0] + inp[0]

        return np.array([x1_dot, x2_dot]) + self.noise[0](x, t)

    @property
    def sys_dim(self) -> SysDim:
        """
        Calculated as mentioned in https://arxiv.org/pdf/2210.01476.pdf
        nz = ny(2nx + 1) # in case of autonomous system
        """
        x_dim, y_dim = 2, 1
        z_dim = y_dim * (2 * x_dim + 1)
        return SysDim(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)
