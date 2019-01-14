from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class HDF5ImageDataset(Dataset):
    """
    PyTorch dataset class for HDF5 files
    """
    def __init__(self, image_dir: Path):
        """
        :param Path image_dir: path to folder with hdf5 files
        """
        super().__init__()
        self.image_files = list(image_dir.glob('*.h*5'))
        self.images = []

        for filepath in tqdm(self.image_files, desc="File"):
            with h5py.File(filepath, 'r') as file:
                img = file['data'][()]
            img = np.expand_dims(img, axis=0)
            img = torch.Tensor(img).div(255).sub(0.5).div(0.5)
            self.images.append(img)

        # check images shape
        self.shape = self.images[0].size()
        if not all([img.size() == self.shape for img in self.images]):
            raise ValueError(f"Not all images have the same shape {self.shape}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]
