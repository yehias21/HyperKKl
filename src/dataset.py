import hydra
from torch.utils.data import Dataset,DataLoader
from omegaconf import DictConfig, OmegaConf
from src.data_preparation import simulate_system_data, simulate_observer_data
from src.simulators.tp import SimTime, ICParam, SigParam, SinParam
from src.simulators.system import Duffing, Observer, SinSignal
from src.utils import gen_dir_time
from typing import Tuple
import torch, pickle as pk, numpy as np, os

# FIXME: 1- Why split the points: PINN Sampling technique
#        2- Is there a reason behind the normalization? better results or theoretical guarantees?
class KKLObserver(Dataset):

    def __init__(self, data:dict, pinn_sample_mode):
        self.data=data

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_index =(self.data['time'].shape[0])
        arr_shape = self.data['states'].shape
        multi_index=np.unravel_index(index,  arr_shape[:-1])
        ss,z_s,y,t= self.data['states'][multi_index], self.data['z_states'][multi_index], self.data['y_out'][multi_index], self.data['time'][index % t_index]

        return ss, z_s, y, t
    def __len__(self) -> int:
        return np.prod(self.data['y_out'].shape)


@hydra.main(config_path="../baselines/config", config_name="duff",  version_base=None)

def load_dataset(cfg: DictConfig,partition:str='train') -> DataLoader:
    OmegaConf.to_container(cfg, resolve=True)
    if partition == 'train':
        sys = Duffing()
        sim_time = SimTime(
            t0=cfg.data.sim_time.start_time,
            tn=cfg.data.sim_time.end_time,
            eps= 0.001
        )
        A, W = [1, 2, 3], [5, 2, 3]
        sig_params = SigParam(sig_name='sinusoids',
                              sig_params=[SinParam(A=3, w=5), SinParam(A=2.1, w=32), SinParam(A=2, w=7),
                                          SinParam(A=6, w=1)])
        inp = SinSignal(sig_params)
        ic_param = ICParam(
            sampler='lhs',
            seed=1212,
            samples=5
        )
        inp.generate_ic(ic_param)
        ic_inp=inp.ic
        print(ic_inp.shape)
        inp_signals = inp.simulate(sim_time)
        print(inp_signals.shape)
        ic_param = ICParam(
            sampler='lhs',
            seed=1212,
            samples=20
        )
        # simulate the system
        states, time = simulate_system_data(sys= sys, solver=cfg.data.solver, sim_time=sim_time, ic_param=ic_param, inp_ic=ic_inp, inp_sys=inp)
        states = np.delete(states, 0, axis=2)
        y_out = sys.get_output(states)
        # Simulate the observer
        import duff_param
        obs = Observer(A=duff_param.OBSERVER_A, B=duff_param.OBSERVER_B)
        z_states = simulate_observer_data(obs=obs, sys=sys, out=y_out,solver=cfg.data.solver,sim_time=sim_time, ic_param=ic_param)

        # Save the data
        # List of file names
        main, sub = gen_dir_time()
        path = '/media/yehias21/DATA/projects/KKL observer/hyperkkl/data'
        pth = os.path.join(path, 'Duffing', main, sub)
        os.makedirs(pth)
        file_names = ['states', 'time', 'y_out', 'z_states', 'input']
        # Iterate through the file names
        for filename, data in zip(file_names, [states, time, y_out, z_states,inp_signals]):
            with open(f'{os.path.join(pth,filename)}.pkl', 'wb') as file:
                pk.dump(data, file)
                print(f"{filename}: {data.shape}")
        # loaded_data = {}
        # # List of file names
        # file_names = ['states', 'time', 'y_out', 'z_states']
        # # Iterate through the file names
        # for filename in file_names:
        #     with open(f'{filename}.pkl', 'rb') as file:
        #         loaded_data[filename] = pk.load(file)
        #         print(f"{filename}: {loaded_data[filename].shape}")
        # # Todo: Dataset forming
        # ds = KKLObserver(data= loaded_data, pinn_sample_mode= 'split_set')
        # data_loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers= 2)
        # return data_loader

if __name__ == "__main__":

    load_dataset()
