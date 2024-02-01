import torch, os, time, random
import numpy as np
from glob import glob


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


def get_files(path, extension='.tif',str_filter=None):
    if str_filter is None:
        return glob(os.path.join(path, '**', f'*{extension}'), recursive=True)
    else:
        raise NotImplementedError
