import copy
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from src.data_loader.data_preparation import simulate_system_data, simulate_observer_data
from typing import Tuple, Optional
import numpy as np

class KKLObserver(Dataset):
    #Todo:
    # 1. Dataset need to be adapted to sequential scheme
    # 2. Dataset need to be adapted to physics trajectories
    # 3. Make the dataset shuffle between trajectories
    def __init__(self, x_states: dict, z_states: dict, time, exo_input: Optional[np.array] = None):
        self.x_states = x_states
        self.z_states = z_states
        self.input_trajectories = exo_input
        self.time = time
        # Check dimensions and set parameters accordingly
        self.inp_ic, self.ic, self.t, self.state_dim = self.x_states['x_regress'].shape

    def __len__(self):
        return self.inp_ic * self.ic * self.t

    def __getitem__(self, idx):
        # Calculate indices for each dimension
        inp_ic_idx = idx // (self.ic * self.t)
        rem = idx % (self.ic * self.t)
        ic_idx = rem // self.t
        t_idx = rem % self.t

        x_sample = {key: self.x_states[key][inp_ic_idx, ic_idx, t_idx, :] for key in self.x_states}
        z_sample = {key: self.z_states[key][inp_ic_idx, ic_idx, t_idx, :] for key in self.z_states}
        input_sample = self.input_trajectories[inp_ic_idx, t_idx] \
            if self.input_trajectories is not None else None

        return {'x_states': x_sample, 'z_states': z_sample, 'exo_input': input_sample,
                'time': self.time[t_idx]}


def load_dataset(cfg: DictConfig, partition: str = 'train') -> DataLoader:
    if partition == 'train':
        #Todo:
        # 1. Input signal adjustement to be (inp_ic, t, inp_dim)
        # 2. Abstract classes adjustments
        # 3. The adjustment of input randomness (what to random and Use randomstate)
        # 4. Implementation of Noise class and adjustments for input, system, observer

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
        observer_states = simulate_observer_data(observer=observer, system=system, y_out=y_out,
                                                 solver=solver, sim_time=sim_time, gen_mode=cfg.gen_mode)
        ###############################################################################################
        match cfg.pinn_sampling:
            # Todo: Implement the pinn_sampling technique
            case 'split_set':
                ph_system = copy.deepcopy(system)
                ph_system.sampler(instantiate(cfg.ph_sampler))
                ph_z_states = ...
                ph_x_states = ...
                x_states = {
                    'x_regress': ...,
                    'x_physics': ...,
                }
                z_states = {
                    'z_regress': ...,
                    'z_physics': ...,
                }
            case 'split_traj':
                ph_z_states = ...
                ph_x_states = ...
                x_states = {
                    'x_regress': ...,
                    'x_physics': ...,
                }
                z_states = {
                    'z_regress': ...,
                    'z_physics': ...,
                }
            case 'no_physics':
                x_states = {
                    'x_regress': states
                }
                z_states = {
                    'z_regress': observer_states
                }
            case _:
                raise ValueError(f"{cfg.pinn_sample_mode} is not a valid sample mode")
        ################################################################################################
        # Todo: Saving the data
        ################################################################################################
        # Todo: split_data into train_set and val_set: time, x_states, z_states, input trajectories - Validation function
        train_set = KKLObserver(x_states=x_states, z_states=z_states, exo_input=input_trajectories, time=time)
        ################################################################################################
        train_loader = DataLoader(train_set, batch_size=cfg.dataloader.batch_size, shuffle=cfg.dataloader.shuffle)
        ################################################################################################
        return train_loader

