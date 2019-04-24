import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from minkowski import compute_minkowski
from utils import fix_random_seed, postprocess_cube

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=10, help="Number of samples to generate & analyze")
    parser.add_argument('--z_dim', type=int, help="Dimension of latent vector z")
    parser.add_argument('--img_factor', type=int, default=1, help="Image size factor")
    parser.add_argument('--cpu', action='store_true', help="Run generation on the CPU")
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment directory')
    parser.add_argument('--checkpoint_name', type=str, help='Name of the net checkpoint')

    args = parser.parse_args()
    seeds = np.random.choice(range(args.num ** 3), size=args.num)
    checkpoint_path = Path('experiments') / args.experiment_name / args.checkpoint_name
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    net_g = torch.load(checkpoint_path).to(device)
    v_list = []
    s_list = []
    b_list = []
    xi_list = []
    for seed in tqdm(seeds, desc="Generate iteration"):
        _ = fix_random_seed(seed)
        noise = torch.randn(1, args.z_dim, args.img_factor, args.img_factor, args.img_factor, device=device)
        cube = net_g(noise).squeeze().detach().cpu()
        cube = cube.mul(0.5).add(0.5).numpy()
        cube = postprocess_cube(cube)
        v, s, b, xi = compute_minkowski(cube)
        v_list.append(v)
        s_list.append(s)
        b_list.append(b)
        xi_list.append(xi)
    df = pd.DataFrame()
    df['V'] = pd.Series(v_list)
    df['S'] = pd.Series(s_list)
    df['B'] = pd.Series(b_list)
    df['Xi'] = pd.Series(xi_list)
    df.to_csv(Path('experiments') / args.experiment_name / 'range_analyze.csv', header=True, index=False)
