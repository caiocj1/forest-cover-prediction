import pandas as pd
import os
import yaml
import numpy as np
from yaml import SafeLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from typing import Optional
from collections import defaultdict

class ForestCoverDataModule(LightningDataModule):
    def __init__(self,
            split_seed: int = 12345,  # split needs to be always the same for correct cross validation
            num_splits: int = 10,
            batch_size: int = 32,
            num_workers: int = 0):
        super().__init__()
        dataset_path = os.getenv('DATASET_PATH')
        self.train_path = os.path.join(dataset_path, 'train.csv')
        self.test_path = os.path.join(dataset_path, 'test-full.csv')

        # Save hyperparemeters
        self.save_hyperparameters(logger=False)

        # Read config file
        self.read_config()

        # Prepare split
        self.kf = KFold(n_splits=self.hparams.num_splits, shuffle=True, random_state=self.hparams.split_seed)

        # Get training set
        read_train_df = pd.read_csv(self.train_path)
        self.train_df = self.format_df(read_train_df)
        self.train_df_input = self.feature_engineering(self.train_df, type='train')

        self.train_mean = self.train_df_input.values.mean(0)
        self.train_std = self.train_df_input.values.std(0)

        # Get test set
        read_test_df = pd.read_csv(self.test_path)
        self.test_ids = read_test_df['Id']
        self.test_df = self.format_df(read_test_df, type='test')
        self.test_df_input = self.feature_engineering(self.test_df, type='test')

        # Fit eventual PCA
        if self.apply_pca:
            self.pca = PCA(self.reduced_dims)
            train_df_normalized = (self.train_df_input.values - self.train_mean) / self.train_std
            self.pca.fit(train_df_normalized)

    def read_config(self):
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params['DatasetParams']

        self.apply_pca = dataset_params['apply_pca']
        self.reduced_dims = dataset_params['reduced_dims']

    def setup(self, stage: str = None, k: int = 0):
        """
        Build data dictionaries for training or prediction.
        :param stage: 'fit' for training, 'predict' for prediction
        :param k: which fold to train on
        :return: None
        """
        assert 0 <= k < self.hparams.num_splits, "incorrect fold number"

        if stage == 'fit':
            # Choose fold to train on
            all_splits = [i for i in self.kf.split(self.train_df_input)]
            train_indexes, val_indexes = all_splits[k]

            train_dict = self.format_X(self.train_df_input.iloc[train_indexes],
                                       self.train_mean,
                                       self.train_std,
                                       type='train',
                                       indexes=train_indexes)
            val_dict = self.format_X(self.train_df_input.iloc[val_indexes],
                                     self.train_mean,
                                     self.train_std,
                                     type='train',
                                     indexes=val_indexes)

            self.data_train, self.data_val = train_dict, val_dict

        elif stage == 'predict':
            predict_dict = self.format_X(self.test_df_input,
                                         self.train_mean,
                                         self.train_std,
                                         type='test')

            self.data_predict = predict_dict

    def train_dataloader(self):
        return DataLoader(dataset=self.data_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.data_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def predict_dataloader(self):
        return DataLoader(dataset=self.data_predict,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          shuffle=False)

    def format_df(self,
                  df: pd.DataFrame,
                  type: str = 'train'):
        """
        Formats the .csv read into an initial dataframe with base features.
        :param df: dataframe read from a csv file
        :param type: whether we are treating a training or test set
        :return: correctly formatted dataframe
        """
        final_df = df.drop(['Id'], axis=1)

        final_df = final_df.drop(['Soil_Type15'], axis=1)

        return final_df

    def feature_engineering(self,
                            df: pd.DataFrame,
                            type: str = 'train'):
        """
        Adds non-trivial features to dataframe, drops target/text/useless columns, so as to prepare the input to the
        deep learning model.
        :param df: formatted dataframe from format_df
        :param type: whether we are treating a training or test set
        :return: correctly formatted input dataframe
        """
        final_df = df.copy()

        if type == 'train':
            final_df = final_df.drop(['Cover_Type'], axis=1)

        # ANGLE TREATMENT
        final_df['Aspect'] = np.cos(final_df['Aspect'] * np.pi / 180.)

        # HORIZONTAL DISTANCES TREATMENT
        final_df['Hor_Total_Mean'] = (final_df['Horizontal_Distance_To_Fire_Points'] + final_df[
            'Horizontal_Distance_To_Roadways'] + final_df['Horizontal_Distance_To_Hydrology']) / 3

        final_df['Hor_Hydr_Fire_Sum'] = final_df['Horizontal_Distance_To_Hydrology'] + final_df[
            'Horizontal_Distance_To_Fire_Points']
        final_df['Hor_Hydr_Road_Sum'] = final_df['Horizontal_Distance_To_Hydrology'] + final_df[
            'Horizontal_Distance_To_Roadways']
        final_df['Hor_Fire_Road_Sum'] = final_df['Horizontal_Distance_To_Fire_Points'] + final_df[
            'Horizontal_Distance_To_Roadways']

        final_df['Hor_Hydr_Fire_Diff'] = np.abs(
            final_df['Horizontal_Distance_To_Hydrology'] - final_df['Horizontal_Distance_To_Fire_Points'])
        final_df['Hor_Hydr_Road_Diff'] = np.abs(
            final_df['Horizontal_Distance_To_Hydrology'] - final_df['Horizontal_Distance_To_Roadways'])
        final_df['Hor_Fire_Road_Diff'] = np.abs(
            final_df['Horizontal_Distance_To_Fire_Points'] - final_df['Horizontal_Distance_To_Roadways'])

        # VERTICAL DISTANCES TREATMENT
        final_df['Ver_Elev_Hydr_Sum'] = final_df['Elevation'] + final_df['Vertical_Distance_To_Hydrology']

        final_df['Ver_Elev_Hydr_Diff'] = np.abs(final_df['Elevation'] - final_df['Vertical_Distance_To_Hydrology'])

        return final_df

    def format_X(self,
                 df: pd.DataFrame,
                 mean: np.ndarray,
                 std: np.ndarray,
                 type: str = 'train',
                 indexes: np.ndarray = None):
        """
        Prepares a dictionary in which to each key is associated a tuple of (input vector, retweet count ground truth),
        from the rows of the dataframe given.
        :param df: correctly formatted input dataframe from feature_engineering
        :param mean: vector with which we normalize the data
        :param std: vector with which we normalize the data
        :param type: whether we are treating a training or test set
        :param indexes: if treating training set, separate train and validation
        :return: correctly dictionary to be passed to DataLoader
        """
        # Get inputs
        X = (df.values - mean) / std

        if hasattr(self, 'pca'):
            X = self.pca.transform(X)

        X = dict(enumerate(X))

        # Get labels
        if type == 'train':
            y = dict(enumerate(self.train_df.iloc[indexes]['Cover_Type'].values - 1))
        else:
            y = dict(enumerate(np.zeros((len(X), ))))

        # Get dict
        final_dict = defaultdict()
        for i in range(len(y)):
            final_dict[i] = (X[i], y[i])

        return final_dict