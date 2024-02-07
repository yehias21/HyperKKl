import numpy as np

# Observer parameters
OBSERVER_A = np.array([-6.5549,  4.6082, -5.2057, 3.3942, 6.0211,
                  -10.9772, -2.3362, -3.7164, -3.9566, -3.7166,
                  -1.9393, -0.2797, -2.7983, -0.8606, -4.8050,
                  -10.5100, -1.0820, -2.6448, -2.1144, -7.0080,
                  -10.1003, -0.5111, 1.0275, 3.1996, -0.3463]).reshape(5,5)

OBSERVER_B = np.ones([5,1])

OBSERVER_STATE_SIZE = 5

# Forwar_Mapper parameters

sim_time = SimTime(
    t0=cfg.data.sim_time.start_time,
    tn=cfg.data.sim_time.end_time,
    eps=0.05
)
A, W = [1, 2, 3], [5, 2, 3]
sig_params = SigParam(sig_name='sinusoids',
                      sig_params=[SinParam(A=3, w=5), SinParam(A=2.1, w=32), SinParam(A=2, w=7),
                                  SinParam(A=6, w=1)])
ic_param = ICParam(
    sampler='lhs',
    seed=1212,
    samples=5
)

ic_param = ICParam(
    sampler='lhs',
    seed=1212,
    samples=20
)