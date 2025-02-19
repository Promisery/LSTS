from .lsts import LSTS


def get_model_cls(name: str):
    name = name.lower()
    if name == 'lsts':
        return LSTS
    raise ValueError(f'Unknown model: {name}')