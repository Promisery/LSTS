from datasets.base import BaseDataset
import numpy as np
import pandas as pd
import glob
import os
import json
import cv2
from scipy.interpolate import interp1d
from tqdm import tqdm


class PUREDataset(BaseDataset):
    def preprocess(self):
        
        data_dirs = glob.glob(os.path.join(self.data_root, '*-*'))
        data_dirs = [d for d in data_dirs if os.path.isdir(d)]
        assert data_dirs, 'PURE Dataset is empty!'

        subjects = [os.path.split(dir)[-1].split('-')[0] for dir in data_dirs]

        subjects = sorted(list(set(subjects)))

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
            subject, record = os.path.split(dir)[-1].split('-')
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
                
            label_file = glob.glob(os.path.join(dir, '*.json'))
            assert len(label_file) == 1, 'More than one label file found!' if len(label_file) > 1 else 'No label file found!'
            label_file = label_file[0]
            with open(label_file, 'r') as f:
                labels = json.load(f)['/FullPackage']
                label_timestamps = [label['Timestamp'] for label in labels]
                waves = [label['Value']['waveform'] for label in labels]
                
            vid_files = np.array(sorted(glob.glob(os.path.join(dir, '*/*.png'))))
            vid_time_stamps = np.array([int(os.path.split(file)[-1].split('.')[0].removeprefix('Image')) for file in vid_files])
            t_start, t_end = min(label_timestamps), max(label_timestamps)
            t_mask = (vid_time_stamps >= t_start) & (vid_time_stamps <= t_end)
            vid_files = vid_files[t_mask]
            vid_time_stamps = vid_time_stamps[t_mask]
            
            interpolator = interp1d(label_timestamps, waves)
            waves = interpolator(vid_time_stamps)
            
            frames = []
            for vid_file in tqdm(vid_files, desc='Loading frames', leave=False):
                img = cv2.imread(vid_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
            
            frames = np.array(frames)
            
            splits = splits.append(self.save(frames, waves, subject, record, split, fold), ignore_index=True)
        
        splits.to_csv(os.path.join('datasets', 'pure_splits.csv'), index=False)
                
            
            