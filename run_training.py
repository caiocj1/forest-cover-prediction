import argparse

from model import ForestCoverModel
from dataset import ForestCoverDataModule

import torch.cuda

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')

    args = parser.parse_args()

    # Model selection
    model = ForestCoverModel()

    # Loggers
    logger = TensorBoardLogger('.', version=args.version)

    lr_monitor = LearningRateMonitor()

    nums_folds = 10
    split_seed = 12345

    for k in range(nums_folds):
        print('Training on split', k, '...')

        data_module = ForestCoverDataModule(
            k=k,
            split_seed=12345,
            num_splits=10,
            reduced_dims=38,
            batch_size=32,
            num_workers=8
        )
        data_module.prepare_data()
        data_module.setup(stage='fit')

        model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{args.version}/checkpoints',
                                     filename='{epoch}-split=%d' % k,
                                     save_top_k=1,
                                     monitor='accuracy_val',
                                     mode='max',
                                     save_weights_only=True)

        # Trainer
        trainer = Trainer(accelerator='auto',
                          devices=1 if torch.cuda.is_available() else None,
                          max_epochs=32,
                          val_check_interval=300,
                          callbacks=[model_ckpt, lr_monitor],
                          logger=logger)

        trainer.fit(model, data_module)
