import argparse
import os
import numpy as np
import pandas as pd
import yaml
import scipy

from models.simple_mlp import SimpleMLPModel
from models.embed_mlp import EmbedMLPModel
from models.embed_sep_mlp import EmbedSeparatelyMLPModel
from dataset import ForestCoverDataModule

import torch.cuda

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

    # Model selection
    model = None
    if args.model == 'mlp':
        model = SimpleMLPModel()
    elif args.model == 'embed':
        model = EmbedMLPModel()
    elif args.model == 'embed_sep':
        model = EmbedSeparatelyMLPModel()

    data_module = ForestCoverDataModule(
        batch_size=32,
        num_workers=8
    )

    trainer = Trainer(accelerator='auto',
                      devices=1 if torch.cuda.is_available() else None)

    results = []
    for ckpt_name in os.listdir(args.weights_path):
        ckpt_path = os.path.join(args.weights_path, ckpt_name)

        # data_module.prepare_data()
        data_module.setup(stage='predict')

        test_results = trainer.predict(model, data_module, ckpt_path=ckpt_path, return_predictions=True)

        predictions = torch.argmax(torch.cat(test_results), dim=1).numpy() + 1
        results.append(predictions)

    final_predictions = scipy.stats.mode(np.array(results)).mode[0]
    submission = pd.DataFrame(data={'Id': data_module.test_ids, 'Cover_Type': final_predictions})

    dataset_path = os.getenv('DATASET_PATH')
    submission_path = os.path.join(dataset_path, 'submission.csv')
    submission.to_csv(submission_path, index=False)

    print('\nSaved submission.csv to DATASET_PATH')
