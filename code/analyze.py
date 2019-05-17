import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from minkowski import compute_minkowski
from models import Generator
from utils import fix_random_seed, postprocess_cube, two_point_correlation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=10, help="Number of samples to generate & analyze")
    parser.add_argument('--z_dim', type=int, help="Dimension of latent vector z")
    parser.add_argument('--img_factor', type=int, default=1, help="Image size factor")
    parser.add_argument('--cpu', action='store_true', help="Run generation on the CPU")
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment directory')
    parser.add_argument('--checkpoint_name', type=str, help='Name of the net checkpoint')

    args = parser.parse_args()
    seeds = np.random.choice(range(14300631), size=args.num)
    size = int(0.9 * (48 + args.img_factor * 16))
    checkpoint_path = Path('experiments') / args.experiment_name / args.checkpoint_name
    covariance_dir = Path('experiments') / args.experiment_name / f'covariance_stats_{size}'
    covariance_dir.mkdir()
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")

    # net_g = torch.load(checkpoint_path).to(device)
    net_g = Generator(
        img_size=64,
        z_dim=args.z_dim,
        num_channels=1,
        num_filters=64,
        num_extra_layers=0
    ).to(device)
    net_g.load_state_dict(torch.load(checkpoint_path))

    data = {
        'V': [],
        'S': [],
        'B': [],
        'Xi': []
    }
    for seed in tqdm(seeds, desc=f"Generate {size}^3, iteration"):
        _ = fix_random_seed(seed)
        noise = torch.randn(1, args.z_dim, args.img_factor, args.img_factor, args.img_factor, device=device)
        cube = net_g(noise).squeeze().detach().cpu()
        cube = cube.mul(0.5).add(0.5).numpy()
        cube = postprocess_cube(cube)
        cube = np.pad(cube, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
        v, s, b, xi = compute_minkowski(cube)
        data['V'].append(v)
        data['S'].append(s)
        data['B'].append(b)
        data['Xi'].append(xi)

        two_point_covariance = {}
        grain_value = cube.max()
        for i, direct in enumerate(["x", "y", "z"]):
            two_point_direct = two_point_correlation(cube, i, var=grain_value)
            two_point_covariance[direct] = two_point_direct
        # phase averaging
        direct_covariances = {}
        for direct in ["x", "y", "z"]:
            direct_covariances[direct] = np.mean(np.mean(two_point_covariance[direct], axis=0), axis=0)
        # covariance storage
        covariance_df = pd.DataFrame(direct_covariances)
        covariance_df.to_csv(covariance_dir / ("seed_" + str(seed) + ".csv"), index=False)

    df = pd.DataFrame(data)
    df.to_csv(Path('experiments') / args.experiment_name / f'seeds_analyze_{size}.csv', index=False)
