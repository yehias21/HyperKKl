import os
from typing import Optional

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
from src.data_loader.data_preparation import simulate_system_data, simulate_kklobserver_data, generate_ph_points
from src.simulators.systems import System


class KKLObserver(Dataset):
    def __init__(self, system: System, observer, x_states: dict, z_states: dict, time,
                 exo_input: Optional[np.array] = None):
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


def load_dataset(cfg: DictConfig, partition: str = 'train') -> [DataLoader, tuple[DataLoader, DataLoader]]:
    """
    :param cfg: configuration file
    :param partition: train or test
    :return: DataLoader
    :description:
    input_trajs: input signal to the system, dimension (inp, t, sig_dim)
    states: system states, dimension (inp, sys_ic, t, x_dim)
    y_out: system output, dimension (inp, sys_ic, t, y_dim)
    ** In case of no input signal, the inp dimension is 1 **
    """
    if partition == 'train':
        if os.path.exists(cfg.get("saved_dataset_path", "")):
            train_set, val_set = torch.load(cfg.saved_dataset_path)
        # Dynamical system initialization
        system = instantiate(cfg.system)
        observer = instantiate(cfg.observer)
        sim_time = instantiate(cfg.sim_time)
        if cfg.get("validation", None):
            if cfg.validation.method == 'long_horizon':
                sim_time.tn += cfg.validation.time
        solver = instantiate(cfg.solver)
        input_trajectories = None
        if 'input_signal' in cfg:
            input_signal = instantiate(cfg.exo_input, _recursive_=False)
            input_trajectories = input_signal.generate_signals(sim_time)

        # simulate the system
        states, time = simulate_system_data(system=system, solver=solver,
                                            sim_time=sim_time, input_data=input_trajectories)
        y_out = system.get_output(states, multi_inp=cfg.multi_inp)
        # Simulate the observer
        observer_states = simulate_kklobserver_data(observer=observer, system=system, y_out=y_out,
                                                    solver=solver, sim_time=sim_time, gen_mode=cfg.gen_mode)
        # dataset creation

        if cfg.get("validation", None):
            match cfg.validation.method:
                case 'long_horizon':
                    val_time = np.arange(cfg.sim_time.tn, cfg.sim_time.tn + cfg.validation.time, cfg.sim_time.eps)
                    t_shape = val_time.shape[0]
                    train_system, val_system = states[:, :, :-t_shape, :], states[:, :, -t_shape:, :]
                    train_observer, val_observer = observer_states[:, :, :-t_shape, :], observer_states[:, :, -t_shape:,
                                                                                        :]
                    train_y_out, val_y_out = y_out[:, :, :-t_shape, :], y_out[:, :, -t_shape:, :]
                    train_input_trajectories, val_input_trajectories = input_trajectories[:, :, :-t_shape,
                                                                       :], input_trajectories[:, :, -t_shape:, :]

                case 'different_configuration':
                    sim_time = instantiate(cfg.validation.sim_time)
                    system.sampler(instantiate(cfg.validation.sampler))
                    observer.sampler(instantiate(cfg.validation.sampler))
                    val_system, val_time = simulate_system_data(system=system, solver=solver, sim_time=sim_time)
                    val_observer = simulate_kklobserver_data(observer=observer, system=system, y_out=y_out,
                                                             solver=solver, sim_time=sim_time)
                    val_y_out = system.get_output(val_system, multi_inp=cfg.multi_inp)
                    val_input_trajectories = None
                    if 'input_signal' in cfg.validation:
                        input_signal = instantiate(cfg.validation.exo_input, _recursive_=False)
                        val_input_trajectories = input_signal.generate_signals(sim_time)

        ph_x_states, ph_z_states = None, None
        if 'pinn_sampling' in cfg:
            sim_time = instantiate(cfg.sim_time)
            ph_x_states, ph_z_states = generate_ph_points(cfg, system, observer, solver, sim_time,
                                                          train_input_trajectories, train_system, train_observer)

        train_set = KKLObserver(system=system, observer=observer,
                                x_states={'x_regress': states, 'x_physics': ph_x_states},
                                z_states={'z_regress': observer_states, 'z_physics': ph_z_states}, time=time,
                                exo_input=train_input_trajectories)
        val_set = KKLObserver(system=system, observer=observer, x_states={'x_regress': val_system},
                              z_states={'z_regress': val_observer}, time=val_time, exo_input=val_input_trajectories)
    # save dataset
    train_loader = DataLoader(train_set, batch_size=cfg.dataloader.batch_size, shuffle=cfg.dataloader.shuffle)
    val_loader = DataLoader(val_set, batch_size=cfg.dataloader.batch_size, shuffle=cfg.dataloader.shuffle)
    return train_loader, val_loader