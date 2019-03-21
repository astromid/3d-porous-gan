import argparse
from pathlib import Path

import torch

from utils import fix_random_seed, save_hdf5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=14300631, help="Random seed")
    parser.add_argument('--z_dim', type=int, help="Dimension of latent vector z")
    parser.add_argument('--img_factor', type=int, default=1, help="Image size factor")
    parser.add_argument('--cpu', action='store_true', help="Run generation on the CPU")
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment directory')
    parser.add_argument('--checkpoint_name', type=str, help='Name of the net checkpoint')

    args = parser.parse_args()
    seed = fix_random_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    checkpoint_path = Path('experiments') / args.experiment_name / args.checkpoint_name
    net_g = torch.load(checkpoint_path).to(device)
    noise = torch.randn(1, args.z_dim, args.img_factor, args.img_factor, args.img_factor, device=device)
    generated_image = net_g(noise).squeeze().detach()
    results_dir = Path('experiments') / args.experiment_name / 'generated'
    results_dir.mkdir(exist_ok=True)
    save_hdf5(generated_image.data, results_dir / f'gen_{args.checkpoint_name}_{seed}_imgf_{args.img_factor}.hdf5')
    print(f"Generated sample with {generated_image.size()}")
