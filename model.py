import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import lightning.pytorch as pl

import gc

from datasets import get_dataset_cls, collate_fn
from models import get_model_cls
from losses import get_loss_cls
from metrics import calculate_hr_and_hrv_metrics


def get_wd_params(module):
    """Weight decay is only applied to a part of the params.
    https://github.com/karpathy/minGPT   

    Args:
        module (Module): torch.nn.Module

    Returns:
        optim_groups: Separated parameters
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.MultiheadAttention)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif 'time_mix' in pn:
                decay.add(fpn)
            else:
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay)) if param_dict[pn].requires_grad]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if param_dict[pn].requires_grad], "weight_decay": 0.0},
    ]
    
    return optim_groups


def get_optimizer_cls(name: str):
    name = name.lower()
    if name == 'adamw':
        return optim.AdamW
    raise ValueError(f'Unknown optimizer: {name}')


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.max_epochs = config['trainer']['max_epochs']
        
        model_cls = get_model_cls(config['model']['name'])
        self.model = model_cls(**config['model']['hparams'])
        
        self.loss_names = [params['name'] for params in config['loss']]
        self.loss_weight_bases = [params['weight'] for params in config['loss']]
        self.loss_weight_exps = [params.get('exp', 1.0) for params in config['loss']]
        self.losses = nn.ModuleList([get_loss_cls(params['name'])() for params in config['loss']])
            
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def on_train_epoch_start(self) -> None:
        self.loss_weights = [base * (exp ** (self.current_epoch / self.max_epochs)) for base, exp in zip(self.loss_weight_bases, self.loss_weight_exps)]
        return super().on_train_epoch_start()
    
    def training_step(self, batch, batch_idx):
        frames, waves, data = batch
        predictions = self(frames)
        loss = 0.
        for loss_name, crit, weight in zip(self.loss_names, self.losses, self.loss_weights):
            loss_value = crit(predictions, waves)
            self.log(f'train/{loss_name}', loss_value, prog_bar=True)
            loss = loss_value * weight + loss
        
        self.log('train/loss', loss, prog_bar=True)
        return loss
    
    def on_test_epoch_start(self):
        self.predictions = {}
        self.ground_truths = {}
        return super().on_test_epoch_start()
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        frames, waves, data = batch
        predictions = self.predict(frames).detach().cpu().numpy()
        batch_size = frames.shape[0]
        for i in range(batch_size):
            
            metadata = data[i]
            subject = metadata['subject']
            record = metadata['record']
            idx = metadata['idx']
            
            if dataloader_idx not in self.predictions:
                self.predictions[dataloader_idx] = {}
                self.ground_truths[dataloader_idx] = {}
            
            if subject not in self.predictions[dataloader_idx]:
                self.predictions[dataloader_idx][subject] = {}
                self.ground_truths[dataloader_idx][subject] = {}
            
            if record not in self.predictions[dataloader_idx][subject]:
                self.predictions[dataloader_idx][subject][record] = {}
                self.ground_truths[dataloader_idx][subject][record] = {}
            
            self.predictions[dataloader_idx][subject][record][idx] = predictions[i]
            self.ground_truths[dataloader_idx][subject][record][idx] = data[i]['waves'].detach().cpu().numpy()
            
        return
    
    def on_test_epoch_end(self):
        
        for dataloader_id in self.predictions.keys():
            predictions = []
            ground_truths = []
            dataloader_predictions = self.predictions[dataloader_id]
            dataloader_ground_truths = self.ground_truths[dataloader_id]
        
            for subject in dataloader_predictions.keys():
                pred_subj = dataloader_predictions[subject]
                gt_subj = dataloader_ground_truths[subject]
                for record in pred_subj.keys():
                    pred_rec = pred_subj[record]
                    gt_rec = gt_subj[record]
                    pred_ = []
                    gt_ = []
                    for i, idx in enumerate(sorted(pred_rec.keys())):
                        pred = pred_rec[idx]
                        gt = gt_rec[idx]
                        if i > 0:
                            pred = pred[-self.config['data']['chunk_interval']:]
                            gt = gt[-self.config['data']['chunk_interval']:]
                        pred_.append(pred)
                        gt_.append(gt)
                    pred_ = np.concatenate(pred_, axis=0)
                    gt_ = np.concatenate(gt_, axis=0)
                    predictions.append(pred_)
                    ground_truths.append(gt_)
            
            metrics = calculate_hr_and_hrv_metrics(predictions, ground_truths, diff='diff' in self.config['data']['wave_type'][0])
            for metric_name, metric_value in metrics.items():
                self.log(f'test/{dataloader_id}/{metric_name}', metric_value, prog_bar='bpm' in metric_name)
        self.predictions = {}
        self.ground_truths = {}
        gc.collect()
        return super().on_test_epoch_end()
    
    def train_dataloader(self):
        train_sets = []
        for args in self.config['data']['train_sets']:
            dataset_cls = get_dataset_cls(args['name'])
            train_sets.append(dataset_cls(**self.config['data']['datasets'][args['name']], split=args['split'], split_idx=self.config['split_idx'], training=True))
        train_set = ConcatDataset(train_sets)
        
        train_loader = DataLoader(
            train_set,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.config['data']['num_workers']>0,
            collate_fn=collate_fn,
        )
        
        return train_loader
   
    def test_dataloader(self):
        test_loaders = []
        for args in self.config['data']['test_sets']:
            dataset_cls = get_dataset_cls(args['name'])
            test_set = dataset_cls(**self.config['data']['datasets'][args['name']], split=args['split'], split_idx=self.config['split_idx'], training=False)
            
            test_loader = DataLoader(
                test_set,
                batch_size=self.config['data']['batch_size'],
                shuffle=False,
                num_workers=self.config['data']['num_workers'],
                pin_memory=False,
                persistent_workers=False,
                collate_fn=collate_fn,
            )
            test_loaders.append(test_loader)
        
        return test_loaders
    
    def configure_optimizers(self):
        optimizer = get_optimizer_cls(self.config['optimizer']['name'])(get_wd_params(self), **self.config['optimizer']['hparams'])
        if 'scheduler' in self.config['optimizer']:
            if self.config['optimizer']['scheduler']['name'] == 'step':
                scheduler = StepLR(optimizer, **self.config['optimizer']['scheduler']['hparams'])
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    },
                }
            elif self.config['optimizer']['scheduler']['name'] == 'onecycle':
                scheduler = OneCycleLR(optimizer, max_lr=self.config['optimizer']['hparams']['lr'], total_steps=self.num_steps, **self.config['optimizer']['scheduler']['hparams'])
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                    },
                }
        return optimizer
    
    @property
    def num_steps(self):
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = dataset_size * self.trainer.max_epochs // (self.trainer.accumulate_grad_batches * num_devices)
        return num_steps