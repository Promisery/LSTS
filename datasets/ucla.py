from datasets.base import BaseDataset
import numpy as np
import pandas as pd
import glob
import os
import cv2
from tqdm import tqdm


class UCLADataset(BaseDataset):
    def preprocess(self):
        data_dirs = sorted(glob.glob(os.path.join(self.data_root, 'subject*')))
                
        subjects = sorted([int(os.path.split(dir)[-1].removeprefix("subject")) for dir in data_dirs])

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
            subject = int(os.path.split(dir)[-1].removeprefix('subject'))
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
            vid_files = glob.glob(os.path.join(dir, f'{os.path.split(dir)[-1].removeprefix("subject")}_*', 'vid.avi'))
            label_files = glob.glob(os.path.join(dir, f'{os.path.split(dir)[-1].removeprefix("subject")}_*', 'ppg.csv'))

            for record, (vid_file, label_file) in enumerate(zip(vid_files, label_files)):
                VidObj = cv2.VideoCapture(vid_file)
                VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
                success, frame = VidObj.read()
                frames = []
                while success:
                    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                    frame = np.asarray(frame)
                    frames.append(frame)
                    success, frame = VidObj.read()
                frames = np.asarray(frames)
                VidObj.release()

                waves = pd.read_csv(label_file, header=None).values.reshape(-1)
                splits = splits.append(self.save(frames, waves, subject, record, split, fold), ignore_index=True)
        
        splits.to_csv(os.path.join('datasets', 'ucla_splits.csv'), index=False)
