import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from retinaface import RetinaFace
from tqdm import tqdm
import cv2
from einops import rearrange
import keras.backend as K

import os
import math
from pathlib import Path

from datasets.utils import BaseTransformer


class BaseDataset(Dataset):
    
    def __init__(
        self,
        name,
        data_root,
        cache_root,
        split,
        split_idx=None,
        transformer=BaseTransformer,
        input_resolution=72,
        training=True,
        rand_augment=False,
        horizontal_flip=True,
        larger_box_coef=1.5,
        detection_freq=180,
        img_height=72,
        img_width=72,
        chunk_length=180,
        chunk_interval=90,
        data_type='normalized',
        wave_type='diff_normalized',
    ):
        assert split_idx is not None or split == 'all'
        split_file = f'./datasets/{name}_splits.csv'
        name = f'{name}-{larger_box_coef}-{detection_freq}-{img_height}-{img_width}-{chunk_length}-{chunk_interval}'
        self.name = name
        self.data_root = data_root
        self.cache_root = cache_root
        self.split = split
        self.transformer = transformer(data_type, wave_type, img_height, img_width, input_resolution, training, rand_augment, horizontal_flip)
        self.larger_box_coef = larger_box_coef
        self.detection_freq = detection_freq
        self.img_height = img_height
        self.img_width = img_width
        self.chunk_length = chunk_length
        self.chunk_interval = chunk_interval
        
        self.data_dir = os.path.join(cache_root, name)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'logs'), exist_ok=True)
        
        if not os.path.isfile(split_file):
            cfg = K.tf.compat.v1.ConfigProto()
            cfg.gpu_options.allow_growth = True
            K.set_session(K.tf.compat.v1.Session(config=cfg))
            self.preprocess()
            K.clear_session()
        self.splits = pd.read_csv(split_file)
        
        if split == 'all':
            pass
        elif split == 'train':
            self.splits = self.splits[self.splits['fold'] != split_idx]
        elif split == 'test':
            self.splits = self.splits[self.splits['fold'] == split_idx]
        else:
            raise ValueError(f'Invalid split: {split}')
            
        self.splits = self.splits['filename'].values
    
    def preprocess(self):
        raise NotImplementedError('BaseDataset not implemented')
    
    def face_detection(self, frame, prior_face_box_coor=None):
        thres = 0.9
        face_zone = []
        while not len(face_zone) and thres > 0.4:
            try:
                resp = RetinaFace.detect_faces(frame, threshold=thres)
                K.clear_session()
                face_zone = [v['facial_area'] for v in resp.values()]
            except:
                thres -= 0.1
                
        for i in range(len(face_zone)):
            face_box_coor = face_zone[i]
            face_box_center = np.array([(face_box_coor[0] + face_box_coor[2]) / 2, (face_box_coor[1] + face_box_coor[3]) / 2])
            len_edge = max(face_box_coor[2]-face_box_coor[0], face_box_coor[3]-face_box_coor[1])

            face_box_coor[0] = face_box_center[0] - len_edge / 2
            face_box_coor[1] = face_box_center[1] - len_edge / 2
            face_box_coor[2] = len_edge
            face_box_coor[3] = len_edge
            
            if self.larger_box_coef > 1:
                face_box_coor[0] = face_box_coor[0] - (self.larger_box_coef - 1.0) / 2 * face_box_coor[2]
                face_box_coor[1] = face_box_coor[1] - (self.larger_box_coef - 1.0) / 2 * face_box_coor[3]
                face_box_coor[2] = self.larger_box_coef * face_box_coor[2]
                face_box_coor[3] = self.larger_box_coef * face_box_coor[3]
            
            face_zone[i] = [int(c) for c in face_box_coor]
           
        if len(face_zone) < 1:
            if prior_face_box_coor is not None:
                face_box_coor = prior_face_box_coor
                print("ERROR: No Face Detected, Using Prior Face Box Coordinates")
            else:
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
                print("ERROR: No Face Detected, Using Whole Frame as Face Box Coordinates")
        elif len(face_zone) >= 2:
            
            zone = [zone for zone in face_zone if ((zone[0] > 0) and (zone[1] > 0) and (zone[0]+zone[2] < frame.shape[1]) and (zone[1] + zone[3]) < frame.shape[0])]
            
            zone_size = [zone[2] * zone[3] for zone in face_zone]
            zone_sort = np.argsort(zone_size)
            if zone_size[zone_sort[-1]] / zone_size[zone_sort[-2]] > 2:
                biggest_zone = np.argmax(zone_size)
                face_box_coor = face_zone[biggest_zone]
            else:
                min_dev = 10000000
                for zone in face_zone:
                    zone_center = [zone[0] + zone[2] // 2, zone[1] + zone[3] // 2]
                    dev = np.abs(zone_center[0] - frame.shape[0] // 2) + np.abs(zone_center[1] - frame.shape[0] // 2)
                    if dev < min_dev:
                        min_dev = dev
                        face_box_coor = zone
            
            frame_with_box = frame.copy()
            for zone in face_zone:
                frame_with_box = cv2.rectangle(
                    frame_with_box,
                    (zone[0], zone[1]),
                    (zone[0]+zone[2], zone[1]+zone[3]),
                    (0, 255, 0),
                    2
                )
            frame_with_box = cv2.rectangle(
                frame_with_box,
                (face_box_coor[0], face_box_coor[1]),
                (face_box_coor[0]+face_box_coor[2],
                 face_box_coor[1]+face_box_coor[3]),
                (255, 0, 0),
            4)
            logs_dir = os.path.join(self.data_dir, 'logs')
            frame_with_box = cv2.cvtColor(frame_with_box, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(logs_dir, f'{len(os.listdir(logs_dir))}.jpg'), frame_with_box)
            
            print("Warning: More than one faces are detected (Only cropping the biggest one.)")
        else:
            face_box_coor = face_zone[0]
        
        return face_box_coor
    
    def face_crop_resize(self, frames, face_box_coor=None):
        # Face Cropping
        if self.detection_freq > 0:
            num_dynamic_det = math.ceil(frames.shape[0] / self.detection_freq)
        else:
            num_dynamic_det = 1
        face_region_all = []
        # Perform face detection by num_dynamic_det" times.
        for idx in range(num_dynamic_det):
            if self.detection_freq > 0:
                face_box_coor = self.face_detection(frames[self.detection_freq * idx], face_box_coor)
                face_region_all.append(face_box_coor)
            else:
                face_region_all.append([0, 0, frames.shape[1], frames.shape[2]])
        face_region_all = np.asarray(face_region_all, dtype='int')

        # Frame Resizing
        resized_frames = np.zeros((frames.shape[0], self.img_height, self.img_height, 3))
        for i in range(0, frames.shape[0]):
            frame = frames[i]
            if self.detection_freq > 0:  # use the (i // self.detection_freq)-th facial region.
                reference_index = i // self.detection_freq
            else:  # use the first region obtrained from the first frame.
                reference_index = 0
            face_region = face_region_all[reference_index]
            frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                    max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            resized_frames[i] = cv2.resize(frame, (self.img_height, self.img_height,), interpolation=cv2.INTER_AREA)
        return resized_frames, face_box_coor
    
    def chunk(self, frames, waves):
        clip_num = (frames.shape[0] - self.chunk_length) // self.chunk_interval + 1
        frames_clips = []
        face_box_coor = None
        for i in tqdm(range(clip_num), desc='Face Detection in Chunks', leave=False):
            clip, face_box_coor = self.face_crop_resize(frames[i*self.chunk_interval: i*self.chunk_interval+self.chunk_length], face_box_coor)
            frames_clips.append(clip)
        waves_clips = [waves[i*self.chunk_interval: i*self.chunk_interval+self.chunk_length] for i in range(clip_num)]
        return frames_clips, waves_clips
    
    def save(self, frames, waves, subject, record, fold):
        
        frames, waves = self.chunk(frames, waves)
        assert len(frames) == len(waves), 'Frames and waves have different lengths!'
        frames = frames
        waves = waves
        splits = []
        
        for i in tqdm(range(len(frames)), desc='Saving', leave=False):
            fi = rearrange(torch.from_numpy(frames[i]).to(torch.uint8), 't h w c -> t c h w').contiguous()
            wi = torch.from_numpy(waves[i]).contiguous().to(torch.float32)
            data = dict(frames=fi, waves=wi)
            filename = f'{subject}-{record}-{i}.pt'
            
            target = Path(os.path.join(self.data_dir, filename))
            
            torch.save(data, target)
            splits.append({'filename': filename, 'fold': fold})
        return splits

    def __len__(self):
        return len(self.splits)
    
    def __getitem__(self, index):
        filename = self.splits[index]
        subject, record, idx = filename.removesuffix('.pt').split('-')
        metadata = {
            'subject': subject,
            'record': record,
            'idx': idx,
            'dataset': self.name
        }
        while True:
            frames, waves, data = self.get_data(metadata, filename)
            if frames.isfinite().all() and waves.isfinite().all():
                break
        return frames, waves, data
    
    @torch.autocast('cpu', enabled=False)
    def get_data(self, metadata, filename):
        metadata.update(torch.load(os.path.join(self.data_dir, filename)))
        return self.transformer(metadata)
        