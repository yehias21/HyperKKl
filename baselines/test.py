# TODO: ASK the Dr about it

def simulate_system_data(system, solver, sim_time, input_trajs: None):
    trajectories = []
    sys_init_cond = system.get_init_cond()
    if input_trajs:
        for input_traj in input_trajs:
            temp_trajs = []
            for ic in sys_init_cond:
                states, time = solver(system.diff_eq, sim_time, ic, input_traj)
                temp_trajs.append(states)
            trajectories.append(temp_trajs)
    else:
        for ic in sys_init_cond:
            states, time = solver(system.diff_eq, sim_time, ic)
            trajectories.append(states)
    return np.array(trajectories), time