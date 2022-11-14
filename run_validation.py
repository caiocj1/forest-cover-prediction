import argparse
import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import torch
import torch.cuda

from models.simple_mlp import SimpleMLPModel
from models.embed_mlp import EmbedMLPModel
from dataset import ForestCoverDataModule

from pytorch_lightning import Trainer

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', '-w', required=True)
    parser.add_argument('--model', '-m', default='mlp')

    args = parser.parse_args()

    # Read config file
    config_path = os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    training_params = params['TrainingParams']

    # Model selection
    model = None
    if args.model == 'mlp':
        model = SimpleMLPModel()
    elif args.model == 'embed':
        model = EmbedMLPModel()

    data_module = ForestCoverDataModule(
        split_seed=training_params['split_seed'],
        num_splits=training_params['num_splits'],
        batch_size=256,
        num_workers=8
    )

    results = []
    for ckpt_name in os.listdir(args.weights_path):
        ckpt_path = os.path.join(args.weights_path, ckpt_name)
        model = model.load_from_checkpoint(ckpt_path)
        model.eval()

        # data_module.prepare_data()
        data_module.setup(stage='fit')
        val_dataloader = data_module.val_dataloader()

        for batch in val_dataloader:
            logits = model(batch)
            loss = torch.nn.functional.cross_entropy(torch.tensor(logits), batch[1], reduction='none')

            plt.hist(loss.detach().numpy())
            plt.show()


