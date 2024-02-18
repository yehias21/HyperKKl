import numpy as np
from src.simulators.solvers import RK4
from src.simulators.system import Duffing, Observer
from src.simulators.tp import ICParam, SimTime
from tqdm import tqdm


def simulate_system_data(sys: Duffing, solver: str, sim_time: SimTime, ic_param: ICParam,
                         inp_ic: np.ndarray = np.array([]), inp_sys=None):
    """ Simulate system """
    match solver.lower():
        case "rk4":
            sys.generate_ic(ic_param)
            sys_ic = sys.ic
            match len(inp_ic):
                case 0:
                    inp_sys = lambda t: 0
                    gen_data = None
                    for x0 in tqdm(sys_ic):
                        states, time = RK4(sys.diff_eq, sim_time, x0, inp_sys)
                        states = states[np.newaxis, :]
                        if gen_data is None:
                            gen_data = states
                        else:
                            gen_data = np.vstack((gen_data, states))
                case _:
                    gen_data = []
                    for inp in inp_ic:
                        gen_temp = []
                        for x0 in sys_ic:
                            inp_sys.set_init_cond(inp)
                            states, time = RK4(sys.diff_eq, sim_time, x0, inp_sys.generate_signal)
                            gen_temp.append(states)
                        gen_data.append(gen_temp)
                    # Convert the list of lists to a numpy array
                    gen_data = np.array(gen_data)
        case _:
            raise ValueError(f"{solver} is not a valid solver")
    return gen_data, time


def simulate_observer_data(obs: Observer, sys: Duffing,out: np.ndarray,solver: str, sim_time: SimTime, ic_param: ICParam):
    """ Simulate observer data out"""
    # FIXME: is this a correct way of generating t0 and thus z0, will the system converge,
    #  in the same time whether it's autonomous or non-autonomous?
    # generate IC and t0_pre
    ic = np.random.rand(ic_param.samples, obs.Z_dim)
    t_neg = obs.calc_pret0(z_max=10 ,e=10e-6)
    # generate x0 for the neg time
    sim_neg = SimTime(
        t0=t_neg,
        tn=0,
        eps=0.05
    )
    neg_states, _ = simulate_system_data(sys, solver, sim_neg, ic_param)
    neg_states = np.delete(neg_states, 0,1)
    neg_out = sys.get_output(neg_states)
    z_init = []
    for z0, y in zip(ic, neg_out):
        z_temp, _ = RK4(obs.diff_eq, sim_neg, z0, inp_gen=y)
        z_init.append(z_temp)

    z_init = np.array(z_init)
    z_init =z_init [:,-1,:]
    z_states= []
    for y_in in out:
        z_states_temp = []
        for z0, y in zip(z_init,y_in):
            z_temp, _ = RK4(obs.diff_eq, sim_time, z0, inp_gen=y)
            z_states_temp.append(z_temp)
        z_states.append(z_states_temp)
    z_states = np.delete(np.array(z_states),0,-2)

    # generate track for

    return z_states
