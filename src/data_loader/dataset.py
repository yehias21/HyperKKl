from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from src.data_loader.data_preparation import simulate_system_data, simulate_observer_data
from typing import Tuple
import torch, numpy as np


class KKLObserver(Dataset):

    def __init__(self, data: dict, pinn_sample_mode):
        self.data = data

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_index = (self.data['time'].shape[0])
        arr_shape = self.data['states'].shape
        multi_index = np.unravel_index(index, arr_shape[:-1])
        ss, z_s, y, t = self.data['states'][multi_index], self.data['z_states'][multi_index], self.data['y_out'][
            multi_index], self.data['time'][index % t_index]

        return ss, z_s, y, t

    def __len__(self) -> int:
        return np.prod(self.data['y_out'].shape)


def get_dataloader(cfg: DictConfig, partition: str = 'train') -> DataLoader:
    pass


def load_dataset(cfg: DictConfig, partition: str = 'train') -> DataLoader:
    if partition == 'train':

        # Dynamical system initialization
        system = instantiate(cfg.system)
        observer = instantiate(cfg.observer)
        sim_time = instantiate(cfg.sim_time)
        solver = instantiate(cfg.solver)
        input_trajectories = None
        if 'input_signal' in cfg:
            # generate the input signal data
            input_signal = instantiate(cfg.input_signal)
            # Generate the input signal
            input_trajectories = input_signal.generate_trajs(sim_time)

        # simulate the system
        states, time = simulate_system_data(system=system, solver=solver,
                                            sim_time=sim_time, input_data=input_trajectories)
        y_out = system.get_output(states)
        # Simulate the observer
        z_states = simulate_observer_data(observer=observer, system=system, y_out=y_out,
                                          solver=solver, sim_time=sim_time)
        print('done')
        ###############################################################################################

if __name__ == "__main__":
    cfg = OmegaConf.load('/media/yehias21/DATA/projects/Work/KKL observer/hyperkkl/baselines/config/duff.yaml')
    load_dataset(cfg.data)
