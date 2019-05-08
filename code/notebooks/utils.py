import random
from pathlib import Path
from typing import NoReturn, Optional
import numba
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
    edge = int(0.1 * cube.shape[0])
    cube = cube[edge:-edge, edge:-edge, edge:-edge]
    # threshold image
    threshold_global_otsu = threshold_otsu(cube)
    cube = (cube >= threshold_global_otsu).astype(np.int32)
    return cube


@numba.jit(nopython=True, parallel=True, nogil=True)
def two_point_correlation(im, dim, var=0):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.

    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction

    var should be set to the pixel value of the pore-space. (Default 0)

    The input image im is expected to be three-dimensional.
    """
    if dim == 0:  # x_direction
        dim_1 = im.shape[2]  # y-axis
        dim_2 = im.shape[1]  # z-axis
        dim_3 = im.shape[0]  # x-axis
    elif dim == 1:  # y-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[1]  # z-axis
        dim_3 = im.shape[2]  # y-axis
    elif dim == 2:  # z-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[2]  # y-axis
        dim_3 = im.shape[1]  # z-axis
    else:
        raise ValueError("Dim error")

    two_point = np.zeros((dim_1, dim_2, dim_3))
    for n1 in range(dim_1):
        for n2 in range(dim_2):
            for r in range(dim_3):
                lmax = dim_3 - r
                for a in range(lmax):
                    if dim == 0:
                        pixel1 = im[a, n2, n1]
                        pixel2 = im[a + r, n2, n1]
                    elif dim == 1:
                        pixel1 = im[n1, n2, a]
                        pixel2 = im[n1, n2, a + r]
                    elif dim == 2:
                        pixel1 = im[n1, a, n2]
                        pixel2 = im[n1, a + r, n2]
                    else:
                        raise ValueError("Dim error")

                    if pixel1 == var and pixel2 == var:
                        two_point[n1, n2, r] += 1
                two_point[n1, n2, r] = two_point[n1, n2, r] / (float(lmax))
    return two_point
