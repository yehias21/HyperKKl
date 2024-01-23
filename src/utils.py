import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os, logging, pickle as pk, time
def seed_everything(seed: int = 17) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_data(cfg:dict , ds: Dataset, data: list[np.ndarray], file_names = list[str])->str:
    #TODO:
    # 1- Change cfg type hint
    # 2- Edit path format: prj_dir +
    #
    tm = time.strftime()
    path = os.path.join(cfg.path,cfg.system)

    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logging.error(f"cannot create directory {path}: {e}")

    for name, dp in zip(file_names, data):
        name = os.path.join(path, name)
        np.savez(name, dp)
        logging.debug(f"File {name.split('/')[-1]} saved!")