import numpy as np
from src.simulators.tp import SimTime
from tqdm import tqdm

def RK4(diff_eq, sim_time: SimTime, x0, inp_gen):
    """
    x0: initial state
    h: step size, delta time or time resolution
    ---
    return states: list of states with shape (t, x_dim)
    """
    states = x0
    time = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
    h = sim_time.eps
    if type(inp_gen) is np.ndarray:
        for ind, t in tqdm(enumerate(time)):
            k1 = diff_eq(t, x0, inp_gen[ind])
            k2 = diff_eq(t + h / 2, x0 + h / 2 * k1, inp_gen[ind])
            k3 = diff_eq(t + h / 2, x0 + h / 2 * k2, inp_gen[ind])
            k4 = diff_eq(t + h, x0 + h * k3, inp_gen[ind])
            # state n+1 at time t+h
            x0 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            states = np.vstack((states, x0))
            x0 = states[ind + 1]
    else:
        for ind, t in tqdm(enumerate(time)):
            k1 = diff_eq(t, x0, inp_gen(t))
            k2 = diff_eq(t + h/2, x0 + h/2*k1, inp_gen(t + h/2))
            k3 = diff_eq(t + h/2, x0 + h/2*k2, inp_gen(t + h/2))
            k4 = diff_eq(t + h, x0 + h*k3, inp_gen(t + h))
            # state n+1 at time t+h
            x0 = x0 + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
            states = np.vstack((states, x0))
            x0 = states[ind+1]
    return states, time