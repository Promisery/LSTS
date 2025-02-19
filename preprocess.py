from datasets import MMPDDataset, UCLADataset, RLAPDataset, PUREDataset


if __name__ == '__main__':
    MMPDDataset('mmpd', 'Path/to/MMPD/dataset', 'Path/to/cache/directory', split='all', training=False, wave_type='normalized', img_height=128, img_width=128)
    UCLADataset('ucla', 'Path/to/UCLA/dataset', 'Path/to/cache/directory', split='all', training=False, wave_type='normalized', img_height=128, img_width=128)
    RLAPDataset('rlap', 'Path/to/RLAP/dataset', 'Path/to/cache/directory', split='all', training=False, wave_type='normalized', img_height=128, img_width=128)
    PUREDataset('pure', 'Path/to/PURE/dataset', 'Path/to/cache/directory', split='all', training=False, wave_type='normalized', img_height=128, img_width=128)