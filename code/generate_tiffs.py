import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from models import Generator
from utils import fix_random_seed, postprocess_cube

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=10, help="Number of samples to generate")
    parser.add_argument('--z_dim', type=int, help="Dimension of latent vector z")
    parser.add_argument('--img_factor', type=int, default=1, help="Image size factor")
    parser.add_argument('--cpu', action='store_true', help="Run generation on the CPU")
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment directory')
    parser.add_argument('--checkpoint_name', type=str, help='Name of the net checkpoint')

    args = parser.parse_args()
    size = int(0.9 * (48 + args.img_factor * 16))
    seeds = np.random.choice(range(14300631), size=args.num)
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    checkpoint_path = Path('experiments') / args.experiment_name / args.checkpoint_name
    net_g = Generator(
        img_size=64,
        z_dim=args.z_dim,
        num_channels=1,
        num_filters=64,
        num_extra_layers=0
    ).to(device)
    net_g.load_state_dict(torch.load(checkpoint_path))

    results_dir = Path('experiments') / args.experiment_name / f'generated_{size}'
    results_dir.mkdir(exist_ok=True)

    for seed in tqdm(seeds, desc=f"Generate {size}^3, iteration"):
        _ = fix_random_seed(seed)
        noise = torch.randn(1, args.z_dim, args.img_factor, args.img_factor, args.img_factor, device=device)
        cube = net_g(noise).squeeze().detach().cpu()
        cube = cube.mul(0.5).add(0.5).numpy()
        cube = postprocess_cube(cube) * 255
        tifffile.imsave(results_dir / f'{seed}.tiff', cube)
