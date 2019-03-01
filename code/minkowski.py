from typing import Tuple

import numba
import numpy as np


@numba.jit
def _update_voxel(rect: np.ndarray) -> Tuple[int, int, int, int]:
    """Computes updates of dn_i values for a given voxel"""
    assert rect.shape == (3, 3, 2)
    if rect[1, 1, 1] == 0:
        return 0, 0, 0, 0
    q = 1 - rect
    dn_3 = 1
    dn_2 = 3 + q[1, 1, 0] + q[1, 2, 1] + q[0, 1, 1]

    dn_1 = 3 + q[1, 1, 0] * q[1, 2, 1] * q[1, 2, 0] + q[1, 1, 0] * q[2, 1, 0] + q[1, 1, 0] * q[1, 0, 0] \
             + q[1, 1, 0] * q[0, 1, 1] * q[0, 1, 0] + q[1, 2, 1] * q[2, 2, 1] + q[0, 1, 1] * 2 \
             + q[0, 2, 1] * q[1, 2, 1] * q[0, 1, 1] + q[1, 2, 1]

    dn_0 = 1 + q[0, 1, 1] + q[1, 2, 1] * q[2, 2, 1] + q[0, 2, 1] * q[1, 2, 1] * q[0, 1, 1] + \
               q[1, 1, 0] * q[2, 1, 0] * q[2, 0, 0] * q[1, 0, 0] + q[0, 1, 0] * q[1, 1, 0] * q[1, 0, 0] * q[0, 0, 0] * \
           q[0, 1, 1] + q[1, 2, 0] * q[2, 2, 0] * q[2, 1, 0] * q[1, 1, 0] * q[1, 2, 1] * q[2, 2, 1] + q[0, 2, 0] * \
           q[1, 2, 0] * q[1, 1, 0] * q[0, 1, 0] * q[0, 2, 1] * q[1, 2, 1] * q[0, 1, 1]
    return dn_3, dn_2, dn_1, dn_0


@numba.jit
def compute_minkowski(cube: np.ndarray) -> Tuple[float, float, float, float]:
    """Computes Minkowski features with regard to whether update dictionary is computed
       Returns V, S, B, Xi features"""
    volume = cube.shape[0] * cube.shape[1] * cube.shape[2]
    cube = np.pad(cube, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    n_0, n_1, n_2, n_3 = 0, 0, 0, 0
    for x in range(1, cube.shape[0] - 1):
        for y in range(1, cube.shape[1] - 1):
            for z in range(1, cube.shape[-1] - 1):
                dn_3, dn_2, dn_1, dn_0 = _update_voxel(cube[x - 1:x + 2, y - 1:y + 2, z - 1:z + 1])
                n_3 += dn_3
                n_2 += dn_2
                n_1 += dn_1
                n_0 += dn_0
    v = n_3 / volume
    s = (-6 * n_3 + 2 * n_2) / volume
    b = (3 * n_3 / 2 - n_2 + n_1 / 2) / volume
    xi = (- n_3 + n_2 - n_1 + n_0) / volume
    return v, s, b, xi
