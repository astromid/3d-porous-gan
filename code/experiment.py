import logging
from pathlib import Path
from datetime import datetime

from train import train_gan

if __name__ == '__main__':
    logger = logging.getLogger()
    experiment_dir = Path('experiments') / datetime.now().strftime('%Y-%m-%d_%H_%M')
    experiment_dir.mkdir()
    train_gan(
        logger=logger,
        experiment_dir=experiment_dir,
        data_dir=Path('data/berea_ti'),
        batch_size=128,
        z_dim=512,
        g_filters=64,
        d_filters=64,
        learning_rate=2e-4,
        beta_1=0.5,
        epochs=100,
        g_iter=3
    )
