import numpy as np
from src.simulators.solvers import RK4
from src.simulators.tp import SimTime, ICParam, SysDim, NoiseParam, SigParam
from src.simulators.sampler import lhs
from typing import Optional


class SinSignal:
    def __init__(self, sig_param: SigParam):
        self.sig_param = sig_param
    def generate_ic(self, ic_param: ICParam) -> list[float]:
        """ Generate initial conditions for the system """
        # TODO: Must specify the sampler
        sample_space = np.array([[-1, 1]])
        match ic_param.sampler.lower():
            case "lhs":
                self._ic = lhs(sample_space, ic_param.seed, ic_param.samples)
            case _:
                raise ValueError(f"{ic_param.sampler} is not a valid sampler")

    @property
    def ic(self) -> np.ndarray:
        return self._ic
    
    #FIXME: search for @property, what it's really doing
    def set_init_cond(self, x0: float) -> float:
        self.init_cond = x0
        
    def generate_signal(self, t: float) -> float:
        """ Generate the input signal """
        input_signal = 0
        for param in self.sig_param.sig_params:
            input_signal += param.A * np.sin(2 * np.pi * param.w * t + self.init_cond * np.pi)
        return float(input_signal)
    
    def simulate(self, sim_time: SimTime) -> np.ndarray:
        time = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
        ini_cond = self.ic
        res = []
        for ic in ini_cond:
            input_signal = 0
            for param in self.sig_param.sig_params:
                input_signal += param.A * np.sin(2 * np.pi * param.w * time + ic * np.pi)
                res.append(input_signal)
        return np.array(res)

class Duffing:
    
    def __init__(self, noise: Optional[NoiseParam] = None) -> None:
        # calculate the z_dim
        self._calc_system_dim()

    def _calc_system_dim(self)-> None:
        """
        Calculated as mentioned in https://arxiv.org/pdf/2210.01476.pdf
        nz = ny(2nx + 1) # in case of autonomous system
        """

        x_dim, y_dim =  2, 1
        z_dim = y_dim * (2*x_dim + 1)
        self._sys_dim = SysDim(x_dim = x_dim, z_dim = z_dim, y_dim = y_dim)

    def generate_ic(self, ic_param: ICParam) -> None:
        """ Generate initial conditions for the system """
        sample_space = np.array([[-1,1],
                                 [-1,1]])
        match ic_param.sampler.lower():
            case "lhs":
                self._ic = lhs(sample_space, ic_param.seed, ic_param.samples)
            case _:
                raise ValueError(f"{ic_param.sampler} is not a valid sampler")  

    def diff_eq(self, t: float, x: list[float], inp: Optional[float] = 0) -> np.ndarray:
        """ System function """
        # TODO: Add the noise, input signal 
        x1_dot = x[1]**3
        x2_dot = - x[0] + inp

        return np.array([x1_dot, x2_dot])
    
    @property
    def ic(self):
        return self._ic  
    
    @property
    def sys_dim(self):
        return self._sys_dim
    
    def get_output(self, states: np.ndarray) -> np.ndarray:
        """ Returns the output of the system """
        C = np.array([1, 0])
        out = np.delete(C*states, 1, axis=-1)
        return out

class Observer:
    def __init__(self, A: np.ndarray, B:np.ndarray) -> None:
        self.A = A
        self.B = B
        self.Z_dim=A.shape[1]
    def diff_eq(self, t: float, z: list[float], y: list[float])-> tuple[float]:
        z_dot = np.matmul(self.A,z) + np.squeeze(self.B*y)
        return z_dot
    
    # function related to the observer
    def calc_pret0(self, z_max:int, e:float):
        w, v = np.linalg.eig(self.A)
        min_ev = np.min(np.abs(np.real(w)))
        kappa = np.linalg.cond(v)
        # s is not mentioned in the paper
        s = np.sqrt(z_max * self.A.shape[0]) # check this: https://en.wikipedia.org/wiki/Norm_(mathematics)#Properties
        t = 1 / min_ev * np.log(e / (kappa * s))
        return t

     
