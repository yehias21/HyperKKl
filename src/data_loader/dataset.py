import copy
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from src.data_loader.data_preparation import simulate_system_data, simulate_kklobserver_data, generate_ph_points
from typing import Optional
import numpy as np
from src.simulators.systems import System


class KKLObserver(Dataset):
    def __init__(self, system: System, observer, x_states: dict, z_states: dict, time, exo_input: Optional[np.array] = None):
        self.system = system
        self.observer = observer
        self.x_states = x_states
        self.z_states = z_states
        self.exo_input = exo_input
        self.time = time
        self.y_out = {key.replace('x', 'y'): self.system.get_output(self.x_states[key]) for key in self.x_states}
        # Check dimensions and set parameters accordingly
        self.inp_ic, self.ic, self.t, _ = (
            self.x_states['x_regress'].shape)  # dimension of the regress must be equal  to dimension of physics

    def __len__(self):
        return self.inp_ic * self.ic * self.t

    def __getitem__(self, idx):
        # Calculate indices for each dimension
        inp_ic_idx = idx // (self.ic * self.t)
        rem = idx % (self.ic * self.t)
        ic_idx = rem // self.t
        t_idx = rem % self.t

        x_sample = {key: self.x_states[key][inp_ic_idx, ic_idx, t_idx, :].astype(np.float32) for key in self.x_states}
        z_sample = {key: self.z_states[key][inp_ic_idx, ic_idx, t_idx, :].astype(np.float32) for key in self.z_states}
        y_out = {key: self.y_out[key][inp_ic_idx, ic_idx, t_idx, :].astype(np.float32) for key in self.y_out}

        if self.exo_input is None:
            return {'x_states': x_sample, 'z_states': z_sample, 'time': self.time[t_idx].astype(np.float32),
                    'y_out': y_out}
        else:
            return {'x_states': x_sample, 'z_states': z_sample,
                    'exo_input': self.exo_input[inp_ic_idx, t_idx].astype(np.float32),
                    'time': self.time[t_idx].astype(np.float32), 'y_out': y_out}


def load_dataset(cfg: DictConfig, partition: str = 'train') -> DataLoader:
    """
    :param cfg: configuration file
    :param partition: train or test
    :return: DataLoader
    :description:
    input_trajs: input signal to the system, dimension (inp, t, sig_dim)
    states: system states, dimension (inp, sys_ic, t, x_dim)
    observer_states: observer states, dimension (inp, sys_ic, t, z_dim)
    y_out: system output, dimension (inp, sys_ic, t, y_dim)
    ** In case of no input signal, the inp dimension is 1 **
    """
    if partition == 'train':
        # Dynamical system initialization
        system = instantiate(cfg.system)
        observer = instantiate(cfg.observer)
        sim_time = instantiate(cfg.sim_time)
        solver = instantiate(cfg.solver)
        input_trajectories = None
        if 'input_signal' in cfg:
            input_signal = instantiate(cfg.input_signal)
            input_trajectories = input_signal.generate_trajs(sim_time)
        # simulate the system
        states, time = simulate_system_data(system=system, solver=solver,
                                            sim_time=sim_time, input_data=input_trajectories)
        y_out = system.get_output(states)
        # Simulate the observer
        observer_states = simulate_kklobserver_data(observer=observer, system=system, y_out=y_out,
                                                    solver=solver, sim_time=sim_time, gen_mode=cfg.gen_mode)
        x_states, z_states = generate_ph_points(cfg, system, observer, solver, sim_time,
                                                input_trajectories, states, observer_states)
################################################################################################
        train_set = KKLObserver(system=system, observer=observer,x_states=x_states, z_states=z_states, exo_input=input_trajectories,
                                time=time)
        train_loader = DataLoader(train_set, batch_size=cfg.dataloader.batch_size, shuffle=cfg.dataloader.shuffle)
        return train_loader

    elif partition == 'test':
        pass


# Todo:
# 1. Validation
# 2. Abstract classes adjustments
# 3. The adjustment of input randomness (what to random and Use randomstate)
# 4. Dataset need to be adapted to sequential scheme and shuffle between trajectories
