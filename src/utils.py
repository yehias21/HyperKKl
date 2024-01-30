import random, time
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


def gen_dir_time() -> tuple[str, str]:
    main_dir = time.strftime("%Y-%m-%d")
    sub_dir = time.strftime("%H-%M-%S")
    return main_dir, sub_dir
