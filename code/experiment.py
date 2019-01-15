from pathlib import Path
import logging
from train import train_gan

if __name__ == '__main__':
    logger = logging.getLogger()
    train_gan(
        logger=logger,
        experiment_dir=Path('../experiments'),
        data_dir=Path('../data/berea_ti'),
        batch_size=2,
        z_dim=8,
        g_filters=4,
        d_filters=4,
        learning_rate=1e-4,
        beta_1=0.5,
        epochs=3
    )
