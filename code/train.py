import pickle
import warnings
from logging import Logger
from pathlib import Path
from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import HDF5ImageDataset
from minkowski import compute_minkowski
from models import Discriminator, Generator
from utils import D_CHECKPOINT_NAME, G_CHECKPOINT_NAME
from utils import fix_random_seed, postprocess_cube

mpl.use('agg')
sns.set()
# smoothing coefficient
ALPHA = 0.98
PRINT_FREQ = 50
# checkpoint filenames prefix
CKPT_PREFIX = 'checkpoint'
FAKE_IMG_FNAME = 'fake_ep{:04d}.hdf5'
REAL_IMG_FNAME = 'real_ep{:04d}.hdf5'
LOGS_FNAME = 'logs.tsv'
PLOT_FNAME = 'plot.png'


def train_gan(
        logger: Logger,
        experiment_dir: Path,
        data_dir: Path,
        batch_size: int,
        z_dim: int,
        g_filters: int,
        d_filters: int,
        learning_rate: float,
        beta_1: float,
        epochs: int,
        saved_g: bool = False,
        saved_d: bool = False,
        seed: Optional[int] = None,
        g_extra_layers: int = 0,
        d_extra_layers: int = 0,
        scheduler: bool = False
) -> None:
    seed = fix_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Train started with seed: {seed}")
    dataset = HDF5ImageDataset(image_dir=data_dir)
    desired_minkowski = pickle.load((data_dir / 'minkowski.pkl').open(mode='rb'))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)
    iterations = epochs * len(loader)
    img_size = dataset.shape[-1]
    num_channels = dataset.shape[0]

    # networks
    net_g = Generator(
        img_size=img_size,
        z_dim=z_dim,
        num_channels=num_channels,
        num_filters=g_filters,
        num_extra_layers=g_extra_layers
    ).to(device)
    net_d = Discriminator(
        img_size=img_size,
        num_channels=num_channels,
        num_filters=d_filters,
        num_extra_layers=d_extra_layers
    ).to(device)
    summary(net_g, (z_dim, 1, 1, 1))
    summary(net_d, (num_channels, img_size, img_size, img_size))

    if saved_g:
        net_g.load_state_dict(torch.load(experiment_dir / G_CHECKPOINT_NAME))
        logger.info("Loaded generator checkpoint")
    if saved_d:
        net_d.load_state_dict(torch.load(experiment_dir / D_CHECKPOINT_NAME))
        logger.info("Loaded discriminator checkpoint")

    # criterion
    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(net_g.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

    patience = int(4500 / len(loader))
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, min_lr=1e-6, verbose=True, patience=patience)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, min_lr=1e-6, verbose=True, patience=patience)

    # labels smoothing
    real_labels = torch.full((batch_size, ), fill_value=0.9, device=device)
    fake_labels = torch.zeros((batch_size, ), device=device)
    fixed_noise = torch.randn(1, z_dim, 1, 1, 1, device=device)

    def step(engine: Engine, batch: torch.Tensor) -> Dict[str, float]:
        """
        Train step function

        :param engine: pytorch ignite train engine
        :param batch: batch to process
        :return batch metrics
        """
        # get batch of fake images from generator
        fake_batch = net_g(torch.randn(batch_size, z_dim, 1, 1, 1, device=device))
        # 1. Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        batch = batch.to(device)
        optimizer_d.zero_grad()
        # train D with real and fake batches
        d_out_real = net_d(batch)
        d_out_fake = net_d(fake_batch.detach())
        loss_d_real = criterion(d_out_real, real_labels)
        loss_d_fake = criterion(d_out_fake, fake_labels)
        # mean probabilities
        p_real = d_out_real.mean().item()
        p_fake = d_out_fake.mean().item()

        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        # 2. Update G network: maximize log(D(G(z)))
        loss_g = None
        p_gen = None
        for _ in range(2):
            fake_batch = net_g(torch.randn(batch_size, z_dim, 1, 1, 1, device=device))
            optimizer_g.zero_grad()
            d_out_fake = net_d(fake_batch)
            loss_g = criterion(d_out_fake, real_labels)
            # mean fake generator probability
            p_gen = d_out_fake.mean().item()
            loss_g.backward()
            optimizer_g.step()

        # minkowski functional measures
        cube = net_g(fixed_noise).detach().squeeze().cpu()
        cube = cube.mul(0.5).add(0.5).numpy()
        cube = postprocess_cube(cube)
        v, s, b, xi = compute_minkowski(cube)
        return {
            'loss_d': loss_d.item(),
            'loss_g': loss_g.item(),
            'p_real': p_real,
            'p_fake': p_fake,
            'p_gen': p_gen,
            'V': v,
            'S': s,
            'B': b,
            'Xi': xi
        }
    # ignite objects
    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        dirname=str(experiment_dir),
        filename_prefix=CKPT_PREFIX,
        save_interval=5,
        n_saved=10,
        require_empty=False
    )

    # attach running average metrics
    monitoring_metrics = ['loss_d', 'loss_g', 'p_real', 'p_fake', 'p_gen', 'V', 'S', 'B', 'Xi']
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['loss_d']).attach(trainer, 'loss_d')
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['loss_g']).attach(trainer, 'loss_g')
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['p_real']).attach(trainer, 'p_real')
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['p_fake']).attach(trainer, 'p_fake')
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['p_gen']).attach(trainer, 'p_gen')
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['V']).attach(trainer, 'V')
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['S']).attach(trainer, 'S')
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['B']).attach(trainer, 'B')
    RunningAverage(alpha=ALPHA, output_transform=lambda x: x['Xi']).attach(trainer, 'Xi')

    # attach progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    @trainer.on(Events.ITERATION_COMPLETED)
    def print_logs(engine):
        if (engine.state.iteration - 1) % PRINT_FREQ == 0:
            fname = experiment_dir / LOGS_FNAME
            columns = ['iter'] + list(engine.state.metrics.keys())
            values = [str(engine.state.iteration)] + [str(round(value, 7)) for value in engine.state.metrics.values()]

            with fname.open(mode='a') as f:
                if f.tell() == 0:
                    print('\t'.join(columns), file=f)
                print('\t'.join(values), file=f)

            message = f"[{engine.state.epoch}/{epochs}][{engine.state.iteration:04d}/{iterations}]"
            for name, value in zip(engine.state.metrics.keys(), engine.state.metrics.values()):
                message += f" | {name}: {value:0.5f}"

            pbar.log_message(message)

    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={'net_g': net_g, 'net_d': net_d})

    @trainer.on(Events.EPOCH_COMPLETED)
    def create_plots(engine):
        df = pd.read_csv(experiment_dir / LOGS_FNAME, delimiter='\t')

        fig_1 = plt.figure(figsize=(18, 12))
        plt.plot(df['iter'], df['loss_d'], label='loss_d', linestyle='dashed')
        plt.plot(df['iter'], df['loss_g'], label='loss_g')
        plt.xlabel('Iteration number')
        plt.legend()
        fig_1.savefig(experiment_dir / ('loss_' + PLOT_FNAME))
        plt.close(fig_1)

        fig_2 = plt.figure(figsize=(18, 12))
        plt.plot(df['iter'], df['p_real'], label='p_real', linestyle='dashed')
        plt.plot(df['iter'], df['p_fake'], label='p_fake', linestyle='dashdot')
        plt.plot(df['iter'], df['p_gen'], label='p_gen')
        plt.xlabel('Iteration number')
        plt.legend()
        fig_2.savefig(experiment_dir / PLOT_FNAME)
        plt.close(fig_2)

        desired_v = [desired_minkowski[0]] * len(df['iter'])
        desired_s = [desired_minkowski[1]] * len(df['iter'])
        desired_b = [desired_minkowski[2]] * len(df['iter'])
        desired_xi = [desired_minkowski[3]] * len(df['iter'])

        fig_3 = plt.figure(figsize=(18, 12))
        plt.plot(df['iter'], df['V'], label='V', color='b')
        plt.plot(df['iter'], desired_v, color='b', linestyle='dashed')
        plt.xlabel('Iteration number')
        plt.ylabel('Minkowski functional V')
        plt.legend()
        fig_3.savefig(experiment_dir / ('minkowski_V_' + PLOT_FNAME))
        plt.close(fig_3)

        fig_4 = plt.figure(figsize=(18, 12))
        plt.plot(df['iter'], df['S'], label='S', color='r')
        plt.plot(df['iter'], desired_s, color='r', linestyle='dashed')
        plt.xlabel('Iteration number')
        plt.ylabel('Minkowski functional S')
        plt.legend()
        fig_4.savefig(experiment_dir / ('minkowski_S_' + PLOT_FNAME))
        plt.close(fig_4)

        fig_5 = plt.figure(figsize=(18, 12))
        plt.plot(df['iter'], df['B'], label='B', color='g')
        plt.plot(df['iter'], desired_b, color='g', linestyle='dashed')
        plt.xlabel('Iteration number')
        plt.ylabel('Minkowski functional B')
        plt.legend()
        fig_5.savefig(experiment_dir / ('minkowski_B_' + PLOT_FNAME))
        plt.close(fig_5)

        fig_6 = plt.figure(figsize=(18, 12))
        plt.plot(df['iter'], df['Xi'], label='Xi', color='y')
        plt.plot(df['iter'], desired_xi, color='y', linestyle='dashed')
        plt.xlabel('Iteration number')
        plt.ylabel('Minkowski functional Xi')
        plt.legend()
        fig_6.savefig(experiment_dir / ('minkowski_Xi_' + PLOT_FNAME))
        plt.close(fig_6)

    if scheduler:
        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler(engine):
            desired_b = desired_minkowski[2]
            desired_xi = desired_minkowski[3]

            current_b = engine.state.metrics['B']
            current_xi = engine.state.metrics['Xi']

            delta = abs(desired_b - current_b) + abs(desired_xi - current_xi)

            scheduler_d.step(delta)
            scheduler_g.step(delta)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            engine.terminate()
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')

            create_plots(engine)
            checkpoint_handler(engine, {
                'net_g_exception': net_g,
                'net_d_exception': net_d
            })
        else:
            raise e

    trainer.run(loader, epochs)
