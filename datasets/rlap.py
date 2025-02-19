from datasets.base import BaseDataset
import numpy as np
import pandas as pd
import glob
import os
import cv2
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from collections import defaultdict


class RLAPDataset(BaseDataset):
    def preprocess(self):
        
        assert os.path.isdir(self.data_root), 'RLAP Dataset is empty!'
        
        df = pd.read_excel('datasets/Dataset Nutrition Label.ods', engine='odf', sheet_name='Videos')
        df = df[df['codec'] == 'RGB']
        df = df[df['fold'] >= 0]
        
        subjects = df['participant'].unique()

        train_subjects = df[df['fold'].apply(lambda fold: fold in [1, 2, 3])]['participant'].unique()
        val_subjects = df[df['fold'].apply(lambda fold: fold in [0])]['participant'].unique()
        test_subjects = df[df['fold'].apply(lambda fold: fold in [4])]['participant'].unique()

        assert set(train_subjects).intersection(val_subjects) == set()
        assert set(train_subjects).intersection(test_subjects) == set()
        assert set(val_subjects).intersection(test_subjects) == set()

        fold1 = df[df['fold'] == 1]['participant'].unique()
        fold2 = df[df['fold'] == 2]['participant'].unique()
        fold3 = df[df['fold'] == 3]['participant'].unique()
        fold4 = df[df['fold'] == 0]['participant'].unique()
        fold5 = df[df['fold'] == 4]['participant'].unique()

        splits = pd.DataFrame(columns=['filename', 'split', 'fold'])
        
        record_cnt = defaultdict(int)
        
        for _, row in tqdm(df.iterrows(), desc='Preprocessing', total=len(df)):
            subject = row['participant']
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
                
            record = record_cnt[subject]
            record_cnt[subject] += 1
            
            label_file = os.path.join(self.data_root, row['BVP'])
            label_file = pd.read_csv(label_file).drop_duplicates(subset=['timestamp'])
            waves = label_file['bvp'].values
            waves_ts = label_file['timestamp'].values
            spline = UnivariateSpline(waves_ts, waves, s=0)
            vid_file_first = os.path.join(self.data_root, row['file'])
            vid_files = sorted(glob.glob(os.path.join(os.path.split(vid_file_first)[0], '*.png')))
            
            frames = []
            for vid_file in tqdm(vid_files, desc='Loading frames', leave=False):
                img = cv2.imread(vid_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
            frames = np.asarray(frames)
            
            ts_file = os.path.join(self.data_root, row['frames_ts'])
            ts_file = pd.read_csv(ts_file)
            ts = ts_file['timestamp'].values
            waves = spline(ts)
            splits = splits.append(self.save(frames, waves, subject, record, split, fold), ignore_index=True)
        
        splits.to_csv(os.path.join('datasets', 'rlap_splits.csv'), index=False)
                
            
            