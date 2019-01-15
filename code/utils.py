import random
from pathlib import Path
from typing import NoReturn, Optional

import h5py
import numpy as np
import torch

G_CHECKPOINT_NAME = 'g.pth'
D_CHECKPOINT_NAME = 'd.pth'


def fix_random_seed(seed: Optional[int] = None) -> int:
    """
    Fix all random seeds
    :param Optional[int] seed: random seed
    :return int: fixed seed
    """
    seed = seed or random.randint(0, 14300631)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def save_hdf5(tensor: torch.Tensor, path: Path) -> NoReturn:
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    :param tensor:
    :param path:
    :return:
    """
    tensor = tensor.cpu()
    numpy_arr = tensor.mul(0.5).add(0.5).mul(255).byte().numpy()
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=numpy_arr, dtype="i8", compression="gzip")
