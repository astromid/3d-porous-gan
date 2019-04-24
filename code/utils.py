import random
from pathlib import Path
from typing import NoReturn, Optional

import h5py
import numpy as np
import torch

from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_otsu

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
    """
    tensor = tensor.cpu()
    numpy_arr = tensor.mul(0.5).add(0.5).numpy()
    numpy_arr = postprocess_cube(numpy_arr) * 255
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=numpy_arr, dtype='u1', compression="gzip")


def postprocess_cube(cube: np.ndarray) -> np.ndarray:
    # singe pixel denoise
    cube = median_filter(cube, size=(3, 3, 3))
    # cut edge noise
    edge = int(0.05 * cube.shape[0])
    cube = cube[edge:-edge, edge:-edge, edge:-edge]
    # threshold image
    threshold_global_otsu = threshold_otsu(cube)
    cube = (cube >= threshold_global_otsu).astype(np.int32)
    return cube
