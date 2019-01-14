import pickle
from copy import copy
from os.path import isfile

import numpy as np


class MinkowskiMeasures:
    def __init__(self, path_to_dict="comb_to_update.p"):
        if isfile(path_to_dict):
            print("Found precomputed dictionary, using fast computation")
            self.dict = pickle.load(open(path_to_dict, "rb"))
            self.fast_flag = True
        else:
            response = "none"
            while response.lower() not in ["yes, y, no, n"]:
                response = input("No dictionary found, generate? (Y/N): ")
            if response in ["yes", "y"]:
                rectangles = self.gen_3d((3, 3, 2))
                self.dict = {}
                for rect in rectangles:
                    self.dict[tuple(rect.flatten())] = self.update_voxel(rect)
                print("Dictionary computed, using fast computation")
                self.fast_flag = True
            else:
                self.fast_flag = False

    @staticmethod
    def pad_cube(cube):
        """Pads cube with zeroes. Returns padded cube"""
        cube_pad = np.zeros((cube.shape[0] + 2, cube.shape[1] + 2, cube.shape[2] + 2))
        cube_pad[1:-1, 1:-1, 1:-1] = cube
        return cube_pad

    def compute_features(self, cube):
        """Computes Minkowski features with regard to whether update dictionary is computed
           Returns V, S, B, Xi features"""
        if self.fast_flag:
            features = self.__compute_features_fast(cube)
        else:
            features = self.__compute_features(cube)
        return features

    def __compute_features(self, cube):
        """Minkowski feature computation without precomputed updates. Works much slower
           Returns V, S, B, Xi features"""
        cube = self.pad_cube(cube)
        n_0, n_1, n_2, n_3 = 0, 0, 0, 0
        for x in range(1, cube.shape[0] - 1):
            for y in range(1, cube.shape[1] - 1):
                for z in range(1, cube.shape[-1] - 1):
                    dn_3, dn_2, dn_1, dn_0 = self.update_voxel(cube[x - 1:x + 2, y - 1:y + 2, z - 1:z + 1])
                    n_3 += dn_3
                    n_2 += dn_2
                    n_1 += dn_1
                    n_0 += dn_0
        V = n_3
        S = -6 * n_3 + 2 * n_2
        B = 3 * n_3 / 2 - n_2 + n_1 / 2
        Xi = - n_3 + n_2 - n_1 + n_0
        return V, S, B, Xi

    def __compute_features_fast(self, cube):
        """Minkowski feature computation with precomputed updates
           Returns V, S, B, Xi features"""
        cube = self.pad_cube(cube)
        n_0, n_1, n_2, n_3 = 0, 0, 0, 0
        for x in range(1, cube.shape[0] - 1):
            for y in range(1, cube.shape[1] - 1):
                for z in range(1, cube.shape[-1] - 1):
                    dn_3, dn_2, dn_1, dn_0 = self.dict[
                        tuple(cube[x - 1:x + 2, y - 1:y + 2, z - 1:z + 1].flatten())]
                    n_3 += dn_3
                    n_2 += dn_2
                    n_1 += dn_1
                    n_0 += dn_0
        V = n_3
        S = -6 * n_3 + 2 * n_2
        B = 3 * n_3 / 2 - n_2 + n_1 / 2
        Xi = - n_3 + n_2 - n_1 + n_0
        return V, S, B, Xi

    @staticmethod
    def gen_3d(size):
        """Generates list of 3D matrices of a given size"""
        template = np.zeros((size, size, size))
        template[:] = np.nan
        cur_list = [template]
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    new_list = []
                    for cube in cur_list:
                        cube_1 = copy(cube)
                        cube_1[x, y, z] = 0
                        cube_2 = copy(cube)
                        cube_2[x, y, z] = 1
                        new_list.append(cube_1)
                        new_list.append(cube_2)
                    cur_list = new_list
        return cur_list

    @staticmethod
    def update_voxel(rect):
        """Computes updates of dn_i values for a given voxel"""
        assert rect.shape == (3, 3, 2)
        if rect[1, 1, 1] == 0:
            return 0, 0, 0, 0
        Q = 1 - rect
        dn_3 = 1
        dn_2 = 3 + Q[1, 1, 0] + Q[1, 2, 1] + Q[0, 1, 1]

        dn_1 = 3 + Q[1, 1, 0] * Q[1, 2, 1] * Q[1, 2, 0] + Q[1, 1, 0] * Q[2, 1, 0] + Q[1, 1, 0] * Q[1, 0, 0] \
                 + Q[1, 1, 0] * Q[0, 1, 1] * Q[0, 1, 0] + Q[1, 2, 1] * Q[2, 2, 1] + Q[0, 1, 1] * 2 \
                 + Q[0, 2, 1] * Q[1, 2, 1] * Q[0, 1, 1] + Q[1, 2, 1]

        dn_0 = 1 + Q[0, 1, 1] + Q[1, 2, 1] * Q[2, 2, 1] + Q[0, 2, 1] * Q[1, 2, 1] * Q[0, 1, 1] + \
                   Q[1, 1, 0] * Q[2, 1, 0] * Q[2, 0, 0] * Q[1, 0, 0] + Q[0, 1, 0] * Q[1, 1, 0] * Q[1, 0, 0] * Q[0, 0, 0] * \
               Q[0, 1, 1] + Q[1, 2, 0] * Q[2, 2, 0] * Q[2, 1, 0] * Q[1, 1, 0] * Q[1, 2, 1] * Q[2, 2, 1] + Q[0, 2, 0] * \
               Q[1, 2, 0] * Q[1, 1, 0] * Q[0, 1, 0] * Q[0, 2, 1] * Q[1, 2, 1] * Q[0, 1, 1]
        return dn_3, dn_2, dn_1, dn_0
