from datasets.base import BaseDataset
import numpy as np
import pandas as pd
import glob
import os
from scipy.io import loadmat
from skimage.transform import resize
from tqdm import tqdm


LIGHT = [
    'LED-low',
    'LED-high',
    'Incandescent',
    'Nature',
]
MOTION = [
    'Stationary',
    'Stationary (after exercise)',
    'Rotation',
    'Talking',
    'Walking',
    'Watching Videos',
]
EXERCISE = [
    'True',
    'False',
]
SKINCOLOR = [
    3,
    4,
    5,
    6,
]
GENDER = [
    'male',
    'female',
]
GLASSER = [
    'True',
    'False',
]
HAIRCOVER = [
    'True',
    'False',
]
MAKEUP = [
    'True',
    'False',
]


class MMPDDataset(BaseDataset):
    def preprocess(self):
        
        data_dirs = glob.glob(os.path.join(self.data_root, '*'))
        assert data_dirs, 'MMPD Dataset is empty!'

        subjects = sorted([os.path.split(dir)[-1].removeprefix("subject").zfill(2) for dir in data_dirs])
        subjects = [int(subject) for subject in subjects]

        train_subjects = subjects[:int(0.6 * len(subjects))]
        val_subjects = subjects[int(0.6 * len(subjects)):int(0.8 * len(subjects))]
        test_subjects = subjects[int(0.8 * len(subjects)):]

        assert set(train_subjects).intersection(val_subjects) == set()
        assert set(train_subjects).intersection(test_subjects) == set()
        assert set(val_subjects).intersection(test_subjects) == set()
        
        fold1 = subjects[:int(0.2 * len(subjects))]
        fold2 = subjects[int(0.2 * len(subjects)):int(0.4 * len(subjects))]
        fold3 = subjects[int(0.4 * len(subjects)):int(0.6 * len(subjects))]
        fold4 = subjects[int(0.6 * len(subjects)):int(0.8 * len(subjects))]
        fold5 = subjects[int(0.8 * len(subjects)):]

        splits = pd.DataFrame(columns=['filename', 'split', 'fold'])
        
        for dir in tqdm(data_dirs, desc='Preprocessing'):
            subject = int(os.path.split(dir)[-1].removeprefix("subject"))
            if subject in train_subjects:
                split = 'train'
            elif subject in val_subjects:
                split = 'val'
            elif subject in test_subjects:
                split = 'test'
            else:
                raise ValueError('Subject not in any split')
            if subject in fold1:
                fold = 1
            elif subject in fold2:
                fold = 2
            elif subject in fold3:
                fold = 3
            elif subject in fold4:
                fold = 4
            elif subject in fold5:
                fold = 5
            else:
                raise ValueError('Subject not in any fold')
            records = glob.glob(os.path.join(dir, '*.mat'))
            records = sorted(records, key=lambda x: int(os.path.split(x)[-1].split('_')[-1].removesuffix('.mat')))
            for record_path in tqdm(records, desc='Loading', leave=False):
                mat = loadmat(record_path)
                
                light = mat['light']
                motion = mat['motion']
                exercise = mat['exercise']
                skin_color = mat['skin_color']
                gender = mat['gender']
                glasser = mat['glasser']
                hair_cover = mat['hair_cover']
                makeup = mat['makeup']
                if (light not in LIGHT) or (motion not in MOTION) or (exercise not in EXERCISE) or \
                    (skin_color not in SKINCOLOR) or (gender not in GENDER) or (glasser not in GLASSER) or \
                    (hair_cover not in HAIRCOVER) or (makeup not in MAKEUP):
                    continue
                
                frames = np.array(mat['video']) * 255
                frames = frames.transpose(1, 2, 0, 3).reshape(320, 240, -1)
                frames = resize(frames, (320, 180))
                frames = frames.reshape(320, 180, 1800, 3).transpose(2, 0, 1, 3)
                waves = np.array(mat['GT_ppg']).T.reshape(-1)
                record = int(os.path.split(record_path)[-1].split('_')[-1].removesuffix('.mat'))
                
                splits = splits.append(self.save(frames, waves, subject, record, split, fold), ignore_index=True)
        
        splits.to_csv(os.path.join('datasets', 'mmpd_splits.csv'), index=False)
                
            
            