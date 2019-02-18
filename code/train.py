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
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import HDF5ImageDataset
from models import Discriminator, Generator
from utils import D_CHECKPOINT_NAME, G_CHECKPOINT_NAME
from utils import fix_random_seed, save_hdf5
from minkowski import MinkowskiMeasures

mpl.use('agg')
sns.set()
# smoothing coefficient
ALPHA = 0.98
PRINT_FREQ = 100
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
        d_extra_layers: int = 0
) -> None:
    """
    Perform GAN training
    :param Logger logger:
    :param Path experiment_dir:
    :param Path data_dir:
    :param int batch_size:
    :param int z_dim:
    :param int g_filters:
    :param int d_filters:
    :param float learning_rate:
    :param float beta_1:
    :param int epochs:
    :param bool saved_g:
    :param bool saved_d:
    :param Optional[int] seed:
    :param int g_extra_layers:
    :param int d_extra_layers:
    :return:
    """
    seed = fix_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Train started with seed: {seed}")
    dataset = HDF5ImageDataset(image_dir=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
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

    # labels smoothing
    real_labels = torch.full((batch_size, ), fill_value=0.9, device=device)
    fake_labels = torch.zeros((batch_size, ), device=device)
    fixed_noise = torch.randn(1, z_dim, 1, 1, 1, device=device)

    # minkowski measurements
    minkowski = MinkowskiMeasures()

    def step(engine: Engine, batch: torch.Tensor) -> Dict[str, float]:
        """
        Train step function

        :param Engine engine: pytorch ignite train engine
        :param torch.Tensor batch: batch to process
        :return Dict[str, float]: batch metrics
        """
        batch = batch.to(device)
        # get batch of fake images from generator
        fake_batch = net_g(torch.randn(batch_size, z_dim, 1, 1, 1, device=device))
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        net_d.zero_grad()
        # train D with real
        d_out_real = net_d(batch)
        loss_d_real = criterion(d_out_real, real_labels)
        # mean real probability
        p_real = d_out_real.mean().item()
        loss_d_real.backward()

        # train D with fake
        d_out_fake = net_d(fake_batch.detach())
        loss_d_fake = criterion(d_out_fake, fake_labels)
        # mean fake probability
        p_fake = d_out_fake.mean().item()
        loss_d_fake.backward()
        # gradient update
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.step()

        # (2) Update G network: maximize log(D(G(z)))
        net_g.zero_grad()
        d_out_fake = net_d(fake_batch)
        loss_g = criterion(d_out_fake, real_labels)
        # mean fake generator probability
        p_gen = d_out_fake.mean().item()
        loss_g.backward()
        optimizer_g.step()

        # minkowski functional measures
        cube = net_g(fixed_noise).detach().squeeze().cpu()
        path = experiment_dir / FAKE_IMG_FNAME.format(engine.state.epoch)
        save_hdf5(cube, path)
        cube = cube.mul(0.5).add(0.5).numpy()
        v, s, b, xi = minkowski.compute_features(cube)
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
        n_saved=5,
        require_empty=False
    )
    timer = Timer(average=True)

    # attach running average metrics
    monitoring_metrics = ['loss_d', 'loss_g', 'p_real', 'p_fake', 'p_gen']
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
            values = [str(engine.state.iteration)] + [str(round(value, 5)) for value in engine.state.metrics.values()]

            with fname.open(mode='a') as f:
                if f.tell() == 0:
                    print('\t'.join(columns), file=f)
                print('\t'.join(values), file=f)

            message = f"[{engine.state.epoch}/{epochs}][{engine.state.iteration:04d}/{iterations}]"
            for name, value in zip(engine.state.metrics.keys(), engine.state.metrics.values()):
                message += f" | {name}: {value:0.4f}"

            pbar.log_message(message)

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_fake_example(engine):
        fake = net_g(fixed_noise)
        path = experiment_dir / FAKE_IMG_FNAME.format(engine.state.epoch)
        save_hdf5(fake.detach(), path)
        # vutils.save_image(fake.detach(), path, normalize=True)

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def save_real_example(engine):
    #     img = engine.state.batch
    #     path = experiment_dir / REAL_IMG_FNAME.format(engine.state.epoch)
    #     save_hdf5(img, path)
    #     # vutils.save_image(img, path, normalize=True)

    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={'net_g': net_g, 'net_d': net_d})

    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]")
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def create_plots(engine):
        df = pd.read_csv(experiment_dir / LOGS_FNAME, delimiter='\t')
        fig_1 = plt.figure(figsize=(18, 12))
        plt.plot(df['iter'], df['loss_d'], label='loss_d')
        plt.plot(df['iter'], df['loss_g'], label='loss_g')
        # _ = df[['loss_d', 'loss_g']].plot()
        plt.xlabel('Iteration number')
        plt.legend()
        fig_1.savefig(experiment_dir / ('loss_' + PLOT_FNAME))
        fig_2 = plt.figure(figsize=(18, 12))
        plt.plot(df['iter'], df['D_x'], label='D_x')
        plt.plot(df['iter'], df['D_G_z1'], label='D_G_z1')
        plt.plot(df['iter'], df['D_G_z2'], label='D_G_z2')
        # _ = df[['D_x', 'D_G_z1', 'D_G_z2']].plot()
        plt.xlabel('Iteration number')
        plt.legend()
        fig_2.savefig(experiment_dir / PLOT_FNAME)

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
