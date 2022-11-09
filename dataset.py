import pandas as pd
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from typing import Optional
from collections import defaultdict

class ForestCoverDataModule(LightningDataModule):
    def __init__(
            self,
            k: int = 0,  # fold number
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,

            reduced_dims: int = None,

            batch_size: int = 32,
            num_workers: int = 0
    ):
        super().__init__()

        dataset_path = os.getenv('DATASET_PATH')
        self.train_path = os.path.join(dataset_path, 'train.csv')
        self.test_path = os.path.join(dataset_path, 'test-full.csv')

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 0 <= self.hparams.k < self.hparams.num_splits, "incorrect fold number"

        # data transformations
        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_node_features() -> int:
        return 4

    @property
    def num_classes() -> int:
        return 2

    def setup(self, stage=None):
        if not self.data_train and not self.data_val:
            train_full = pd.read_csv(self.train_path)
            train_full = train_full.drop(['Id', 'Soil_Type15'], axis=1)
            self.train_mean = train_full.values.mean(0)
            self.train_std = train_full.values.std(0)

            if stage == 'fit':
                # choose fold to train on
                kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)
                all_splits = [k for k in kf.split(train_full)]
                train_indexes, val_indexes = all_splits[self.hparams.k]

                train_df = train_full.iloc[train_indexes]
                val_df = train_full.iloc[val_indexes]

                # Get inputs
                train_X = (train_df.values - self.train_mean) / self.train_std
                val_X = (val_df.values - self.train_mean) / self.train_std

                if self.hparams.reduced_dims:
                    pca = PCA(self.hparams.reduced_dims)

                    train_X = pca.fit_transform(train_X)
                    val_X = pca.transform(val_X)

                train_X = dict(enumerate(train_X))
                val_X = dict(enumerate(val_X))

                # Get labels
                train_y = dict(enumerate(train_df['Cover_Type'].values - 1))
                val_y = dict(enumerate(val_df['Cover_Type'].values - 1))

                # Get dict
                train_dict = defaultdict()
                val_dict = defaultdict()
                for i in range(len(train_y)):
                    train_dict[i] = (train_X[i], train_y[i])
                for i in range(len(val_y)):
                    val_dict[i] = (val_X[i], val_y[i])

                self.data_train, self.data_val = train_dict, val_dict

            elif stage == 'test':
                test_full = pd.read_csv(self.test_path)
                test_full = test_full.drop(['Id', 'Soil_Type15'], axis=1)
                pass

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)