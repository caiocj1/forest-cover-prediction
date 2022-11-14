import torch.optim
from pytorch_lightning import LightningModule
import torch.nn as nn
import os
import yaml
from yaml import SafeLoader
from collections import OrderedDict

class EmbedSeparatelyMLPModel(LightningModule):

    def __init__(self):
        super(EmbedSeparatelyMLPModel, self).__init__()
        self.read_config()
        self.build_model()

    def read_config(self):
        config_path = os.path.join(os.getcwd(), './config.yaml')
        with open(config_path) as f:
            params = yaml.load(f, Loader=SafeLoader)
        dataset_params = params['DatasetParams']
        model_params = params['ModelParams']

        self.apply_pca = dataset_params['apply_pca']
        self.reduced_dims = dataset_params['reduced_dims']

        self.layer_width = model_params['layer_width']
        self.num_layers = model_params['num_layers']
        self.dropout = model_params['dropout']
        self.embed_dims = model_params['embed_dims']

    def build_model(self):
        assert not self.apply_pca, 'Must disable pca'

        self.embed_soil_type = nn.Sequential(
            nn.Linear(39, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.embed_dims)
        )

        self.embed_wilderness_area = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.input = nn.Linear(10 + self.embed_dims + 1, self.layer_width)

        hidden_layers_dict = OrderedDict()
        for i in range(self.num_layers - 2):
            hidden_layers_dict['layer' + str(i + 1)] = nn.Linear(self.layer_width // (2**i), self.layer_width // (2**(i+1)))
            hidden_layers_dict['relu' + str(i + 1)] = nn.ReLU()
            if self.dropout:
                hidden_layers_dict['dropout' + str(i + 1)] = nn.Dropout(p=0.25)
        self.hidden_layers = nn.Sequential(hidden_layers_dict)

        self.output = nn.Linear(self.layer_width // (2 ** (self.num_layers - 2)), 7)

        self.relu = nn.ReLU()

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'train')
        self.log('loss_train', loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        self.log_metrics(metrics, 'val')
        self.log('loss_val', loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)

        loss = loss.mean()
        return loss

    def _shared_step(self, batch):
        logits = self.forward(batch)

        loss = self.calc_loss(logits, batch[1])

        metrics = self.calc_metrics(logits, batch[1])

        return loss, metrics

    def forward(self, batch):
        soil_type = batch[0][:, -39:].float()
        soil_type_embedded = self.embed_soil_type(soil_type)
        wilderness_area = batch[0][:, -43:-39].float()
        wilderness_area_embedded = self.embed_wilderness_area(wilderness_area)
        nums = batch[0][:, :-43].float()
        input = torch.cat([nums, wilderness_area_embedded, soil_type_embedded], axis=1)

        encoding = self.input(input)
        encoding = self.hidden_layers(self.relu(encoding))
        logits = self.output(encoding)

        return logits

    def calc_loss(self, prediction, target):
        ce_loss = nn.CrossEntropyLoss(reduction='none')

        loss = ce_loss(prediction, target)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [128], gamma=0.1)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [64, 128, 192], gamma=0.5)

        return [optimizer], []

    def calc_metrics(self, logits, target):
        metrics = {}

        prediction = torch.argmax(logits, dim=1)
        batch_size = len(logits)

        metrics['accuracy'] = (prediction == target).sum() / batch_size

        return metrics

    def log_metrics(self, metrics: dict, type: str):
        on_step = True if type == 'train' else False

        for key in metrics:
            self.log(key + '_' + type, metrics[key], on_step=on_step, on_epoch=True, logger=True)
