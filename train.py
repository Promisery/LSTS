import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint

from argparse import ArgumentParser
import yaml
import os
from functools import partial

from model import Model

def get_callback_cls(name: str):
    name = name.lower()
    if name == 'devicestatsmonitor':
        return partial(
            DeviceStatsMonitor,
            cpu_stats=True,
        )
    elif name == 'learningratemonitor':
        return LearningRateMonitor
    elif name == 'modelcheckpoint':
        return partial(
            ModelCheckpoint,
            save_last=True,
            every_n_epochs=0,
        )
    elif name == 'valckpt':
        return partial(
            ModelCheckpoint,
            save_top_k=1,
            monitor='val/0/bpm/RMSE',
        )
    raise ValueError(f'Unknown callback: {name}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--split_idx', type=int, required=True, help='Index of split in 5-fold cross validation')
    args = parser.parse_args()
    
    run_name = os.path.split(args.config)[-1].split('.')[0]
    proj_name = run_name.split('_')[-1]
        
    save_dir = os.path.join('logs', run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['split_idx'] = args.split_idx
    
    pl.seed_everything(config['seed'], workers=True)
        
    model = Model(config)
    
    logger = WandbLogger(
        name=run_name+f'_fold{args.split_idx}',
        save_dir=save_dir,
        project=proj_name,
        log_model=True,
    )
    
    callbacks = []
    for name in config['trainer']['callbacks']:
        callbacks.append(get_callback_cls(name)())
            
    trainer = pl.Trainer(
        precision='16-mixed',
        max_epochs=config['trainer']['max_epochs'],
        deterministic='warn',
        logger=logger,
        callbacks=callbacks,
        default_root_dir=save_dir,
    )
    
    trainer.fit(model)
    
    trainer.test(model, ckpt_path='last')
    