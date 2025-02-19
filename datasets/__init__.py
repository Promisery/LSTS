import torch
from .mmpd import MMPDDataset
from .ucla import UCLADataset
from .rlap import RLAPDataset
from .pure import PUREDataset


def get_dataset_cls(name: str):
    name = name.lower()
    if name == 'mmpd':
        return MMPDDataset
    elif name == 'ucla':
        return UCLADataset
    elif name == 'rlap':
        return RLAPDataset
    elif name == 'pure':
        return PUREDataset
    raise ValueError(f'Unknown dataset: {name}')


def collate_fn(batch):
    frames, waves, data = zip(*batch)
    frames = torch.stack(frames, dim=0)
    waves = torch.stack(waves, dim=0)
    return frames, waves, data