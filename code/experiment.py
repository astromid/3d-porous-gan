from pathlib import Path

from train import train_gan

if __name__ == '__main__':
    train_gan(
        logger=None,
        experiment_dir=Path('../experiments'),
        data_dir=Path('../data/berea_ti'),
        batch_size=1,
        z_dim=16,
        g_filters=8,
        d_filters=8,
        learning_rate=1e-4,
        beta_1=0.5,
        epochs=1
    )
