import argparse
import os
import yaml

from models.simple_mlp import SimpleMLPModel
from models.embed_mlp import EmbedMLPModel
from models.embed_sep_mlp import EmbedSeparatelyMLPModel
from dataset import ForestCoverDataModule

import torch.cuda

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v')
    parser.add_argument('--model', '-m', default='mlp')

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    training_params = params['TrainingParams']

    # Initialize data module
    data_module = ForestCoverDataModule(
        split_seed=training_params['split_seed'],
        num_splits=training_params['num_splits'],
        batch_size=32,
        num_workers=8)

    for k in range(training_params['num_splits']):
        print('Training on split', k, '...')

        # Initialize new model and setup data module
        model = None
        if args.model == 'mlp':
            model = SimpleMLPModel()
        elif args.model == 'embed':
            model = EmbedMLPModel()
        elif args.model == 'embed_sep':
            model = EmbedSeparatelyMLPModel()
        data_module.setup(stage='fit', k=k)

        # Loggers and checkpoints
        version = args.version + '_split=' + str(k)
        logger = TensorBoardLogger('.', version=version)
        model_ckpt = ModelCheckpoint(dirpath=f'lightning_logs/{args.version}_CV/checkpoints',
                                     filename='{epoch}-split=%d' % k,
                                     save_top_k=1,
                                     monitor='accuracy_val',
                                     mode='max',
                                     save_weights_only=True)
        lr_monitor = LearningRateMonitor()

        # Trainer
        trainer = Trainer(accelerator='auto',
                          devices=1 if torch.cuda.is_available() else None,
                          max_epochs=256,
                          val_check_interval=300,
                          callbacks=[model_ckpt, lr_monitor],
                          logger=logger)
        trainer.fit(model, data_module)
