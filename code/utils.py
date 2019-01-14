import random
from typing import Optional

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
