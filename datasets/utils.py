import numpy as np
from scipy.signal import find_peaks
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from einops import rearrange


SIGNAL_TYPE = ['raw', 'normalized', 'diff', 'diff_normalized']
numpy_to_torch_dtype_dict = {
    np.bool       : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

class BaseTransformer:
    def __init__(
        self,
        data_type,
        wave_type,
        img_height,
        img_width,
        input_resolution=128,
        training=True,
        rand_augment=False,
        horizontal_flip=True,
        resized_crop=False,
        affine=False,
        color_jitter=False,
        auto_contrast=False,
        adjust_sharpness=False,
        eps=1e-8,
    ):
        data_type = data_type if isinstance(data_type, list) else [data_type]
        wave_type = wave_type if isinstance(wave_type, list) else [wave_type]
        self.data_type = data_type
        self.wave_type = wave_type
        for d in data_type + wave_type:
            assert d in SIGNAL_TYPE, f'Unknown signal type {d}'
    
        transform = []
        
        if input_resolution > 0 and ((input_resolution != img_height) or (input_resolution != img_width)):
            transform.append(transforms.Resize(input_resolution))
        
        if training:
            if rand_augment:
                transform.append(transforms.RandAugment(num_ops=2, magnitude=9))
            if horizontal_flip:
                transform.append(transforms.RandomHorizontalFlip())
            if resized_crop:
                transform.append(transforms.RandomResizedCrop((input_resolution, input_resolution), scale=(0.8, 1.0), ratio=(0.8, 1.2)))
            if affine:
                transform.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(10, 10, 10, 10)))
            if color_jitter:
                transform.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
            if auto_contrast:
                transform.append(transforms.RandomAutocontrast())
            if adjust_sharpness:
                transform.append(transforms.RandomAdjustSharpness(2.0))
        
        self.vision_transforms = nn.Sequential(*transform)
        
        self.eps = eps
        
        
    def transform(self, series, types):
        results = []
        for t in types:
            if t == 'raw':
                results.append(series)          
            elif t == 'normalized':
                if series.ndim > 1:
                    results.append((series - 0.5) * 2)
                else:
                    results.append(self.norm_bvp(series))
            elif 'diff' in t:                
                diff_series = series.clone()
                diff_series[:-1] = series[1:] - series[:-1]
                if 'normalized' in t:
                    if series.ndim > 1:
                        diff_series[:-1] = diff_series[:-1] / (series[1:] + series[:-1] + self.eps)
                    # else:
                    diff_series[:-1] = diff_series[:-1] / (diff_series[:-1].std() + self.eps)
                diff_series[-1:].fill_(0.)
                # diff_series.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                results.append(diff_series)
        return torch.cat(results, dim=1) if series.ndim > 1 else torch.stack(results, dim=-1)
    
    def __call__(self, data):
        frames_raw = data['frames']
        waves_raw = data['waves']
        
        frames_raw = self.vision_transforms(frames_raw) / 255.
                
        frames = self.transform(frames_raw, self.data_type)
        waves = self.transform(waves_raw, self.wave_type)
        
        return frames, waves, data
    
    def norm_bvp(self, bvp, order=1, dtype=None):
        dtype = dtype or bvp.dtype
        if not isinstance(bvp, np.ndarray):
            bvp = bvp.numpy()
        else:
            dtype = numpy_to_torch_dtype_dict[dtype.type]
        order -= 1
        bvp = (bvp-np.mean(bvp))/np.std(bvp)
        prominence = (1.5*np.std(bvp), None)
        peaks = np.sort(np.concatenate([find_peaks(bvp, prominence=prominence)[0], find_peaks(-bvp, prominence=prominence)[0]]))
        bvp = np.concatenate((bvp, bvp[-1:]))
        bvp = np.concatenate([((x-np.mean(x))/np.abs(x[0]-x[-1]))[:-1] for x in (bvp[a:b] for a, b in zip(np.concatenate([(0,), peaks]), np.concatenate([peaks+1, (len(bvp),)])))])
        bvp = np.nan_to_num(bvp, nan=0.0, posinf=0.0, neginf=0.0)
        if order == 0:
            return torch.tensor(np.clip(bvp, a_max=0.5, a_min=-0.5), dtype=dtype)
        else:
            return self.norm_bvp(bvp, order)
