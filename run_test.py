import argparse
import os
import yaml

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
    parser.add_argument('--weights_path', '-w', required=True)

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    # Model selection
    model = ForestCoverModel()

    data_module = ForestCoverDataModule(
        batch_size=1,
        num_workers=8
    )

    results = []
    for ckpt_name in os.listdir(args.weights_path):
        ckpt_path = os.path.join(args.weights_path, ckpt_name)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        print('loaded weights')
        model.eval()

        # data_module.prepare_data()
        data_module.setup(stage='test')
        test_dict = data_module.data_test

        for key in test_dict:
            pass

