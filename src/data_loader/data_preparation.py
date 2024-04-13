from typing import Optional
import numpy as np
from src.simulators.systems import System
from src.simulators.estimators import KKLObserver

from src.simulators.types import SimTime
from tqdm import tqdm
from hydra.utils import instantiate


def simulate_system_data(system, solver, sim_time, input_data: Optional[np.ndarray] = None):
    """
     input_trajs: input signal to the system, dimension (n, t, sig_dim)
     :returns: system states, dimension (inp, sys_ic, t, x_dim)
     """
    trajectories = []
    if input_data is not None:
        for input_traj in tqdm(input_data, desc="Input Loop", position=0):
            temp_trajs = []
            for ic in tqdm(system.ic, desc="system Loop", position=1, leave=False):
                states, _ = solver(system.diff_eq, sim_time, ic, input_traj)
                temp_trajs.append(states)
            trajectories.append(temp_trajs)
    else:
        temp_trajs = []
        for ic in tqdm(system.ic):
            states, _ = solver(system.diff_eq, sim_time, ic)
            temp_trajs.append(states)
        trajectories.append(temp_trajs)
    # drop the initial conditions from system states
    trajectories = np.delete(np.array(trajectories), 0, -2)
    return trajectories, np.arange(sim_time.t0, sim_time.tn, sim_time.eps)


def simulate_kklobserver_data(observer: KKLObserver, system: System, y_out: np.ndarray,
                              solver, sim_time: SimTime, gen_mode='forward'):
    """ Simulate observer data out"""
    # checks
    assert gen_mode in ['forward', 'backward'], "gen_mode should be either 'forward' or 'backward'"
    # backward distinguishability
    t_neg = observer.calc_pret0()
    sim_neg = SimTime(sim_time.t0, sim_time.t0 + t_neg, sim_time.eps) if gen_mode == 'backward' else SimTime(
        t_neg + sim_time.t0, sim_time.t0,
        sim_time.eps)
    # simulate the system in the negative time, with same initial condition of forward time
    neg_states, _ = simulate_system_data(system, solver, sim_neg)
    neg_out = system.get_output(neg_states)
    neg_out = np.flip(neg_out, axis=-2) if gen_mode == 'backward' else neg_out
    # converge to the initial condition of the observer so that Z0 = T(X0)
    neg_out = np.squeeze(neg_out, 0)
    z_init = []
    for z0, y in zip(observer.ic, neg_out):
        z_temp, _ = solver(observer.diff_eq, sim_neg, z0, exogenous_input=y)
        z_init.append(z_temp)
    z_init = np.array(z_init)
    z_init = z_init[:, -1, :]
    assert z_init.shape[0] == y_out.shape[
        -3], "The initial conditions for the observer should be the same as the output"

    # simulate the observer
    z_states = []
    # First loop over the input signal
    for y_in in y_out:
        z_states_temp = []
        # Inner loop for each initial condition Z0
        for z0, y_traj in zip(z_init, y_in):
            z_temp, _ = solver(observer.diff_eq, sim_time, z0, exogenous_input=y_traj)
            z_states_temp.append(z_temp)
        z_states.append(z_states_temp)

    # drop the initial conditions from system states
    z_states = np.delete(np.array(z_states), 0, -2)
    return z_states


def generate_ph_points(cfg, system, observer, solver, sim_time, input_trajectories, states, observer_states):
    match cfg.pinn_sampling:
        case 'split_set':
            system.sampler(instantiate(cfg.ph_sampler))
            ph_x_states, _ = simulate_system_data(system=system, solver=solver,
                                                  sim_time=sim_time, input_data=input_trajectories)
            ph_out = system.get_output(ph_x_states)
            ph_observer_states = simulate_kklobserver_data(observer=observer, system=system, y_out=ph_out,
                                                           solver=solver, sim_time=sim_time, gen_mode=cfg.gen_mode)
            x_states = {
                'x_regress': states,
                'x_physics': ph_x_states
            }
            z_states = {
                'z_regress': observer_states,
                'z_physics': ph_observer_states
            }
        case 'split_traj':
            if states.shape[1] == 1:
                raise ValueError("Cannot split along the first dimension if its size is 1")

            x_states = {
                'x_regress': states[:, ::2],
                'x_physics': states[:, 1::2],
            }

            z_states = {
                'z_regress': observer_states[:, ::2],
                'z_physics': observer_states[:, 1::2],
            }

        case 'no_physics':
            x_states = {
                'x_regress': states,
            }
            z_states = {
                'z_regress': observer_states,
            }
        case _:
            raise ValueError(f"{cfg.pinn_sample_mode} is not a valid sample mode")
    return x_states, z_states
