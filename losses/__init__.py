from .corr import NegPearson


def get_loss_cls(name: str):
    name = name.lower()
    if name == 'negpearson':
        return NegPearson
    raise ValueError(f'Unknown loss: {name}')