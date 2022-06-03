import random
import torch
import cv2
import numpy as np
from scipy.ndimage import interpolation
import gzip

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, volume_ids, input_dim=128, half_precision=False, device='cpu', loss_approach='area', rotate_volume=False):
        'Initialization'
        self.volume_ids = volume_ids
        self.input_dim=input_dim
        self.half_precision = half_precision
        self.loss_approach=loss_approach
        self.rotate_volume=rotate_volume

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.volume_ids)

    def load_X(self, volume_id, input_dim=128, rotate_by=None, on_axes=(-2,-1)):
        pass

    def load_y(self, volume_id, input_dim=128, rotate_by=None, on_axes=(-2,-1)):
        pass

    def set_input_dim(self, randomize=True, permute_from=[], input_dim=None):
        if randomize:
            self.input_dim = random.choice(permute_from)
        else:
            self.input_dim = input_dim

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        volume_id = self.volume_ids[index]
        if self.rotate_volume:
            if random.random() >=0.5:
                angle = random.choice([-90, 90, 180])
                on_axes = sorted(random.sample([-1,-2,-3], 2))
                X, y = self.load_X(volume_id, input_dim=self.input_dim, rotate_by=angle), self.load_y(volume_id, input_dim=self.input_dim, rotate_by=angle, on_axes=on_axes)
                return X, y
                
        X, y = self.load_X(volume_id, input_dim=self.input_dim), self.load_y(volume_id, input_dim=self.input_dim)

        return X, y


class BRATSDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, volume_ids, input_dim=128, half_precision=False, device='cpu', loss_approach='area', rotate_volume=False):
        'Initialization'
        super(BRATSDataset, self).__init__(volume_ids, input_dim=128, half_precision=False, device='cpu',loss_approach=loss_approach, rotate_volume=False)

    def load_X(self, volume_id, input_dim=128, rotate_by=None, on_axes=(-2,-1)):
        with gzip.GzipFile('data/BRATS-3D/volumes/{}.npy.gz'.format(volume_id), 'r') as f:
            volume = np.load(f)
        if rotate_by:
            volume = rotate(volume, axes=(-2,-1), angle=rotate_by)
        # BRATS
        original_dim = volume.shape[1]
        reshaped_volume = []
        for i in range(volume.shape[0]):
            reshaped_volume += [interpolation.zoom(volume[i], zoom=input_dim/original_dim, order=0)]
        return np.asarray(reshaped_volume, dtype=np.float32)

    def load_y(self, volume_id, input_dim=128, rotate_by=None, on_axes=(-2,-1)):
        with gzip.GzipFile('data/BRATS-3D/masks/{}.npy.gz'.format(volume_id), 'r') as f:
            mask = np.load(f)
            
        if rotate_by:
            mask = interpolation.rotate(mask, axes=(-2,-1), angle=rotate_by)
            
        if self.loss_approach == 'ce':
            return mask.astype(int)
        original_dim = mask.shape[1]

        mask_0 = (mask == 0).astype(int)
        mask_1 = (mask == 1).astype(int)
        mask_2 = (mask == 2).astype(int)
        mask_3 = (mask == 3).astype(int)

        mask_WT = 1 - mask_0
        mask_TC = np.logical_or(mask_2, mask_3)

        return np.asarray([mask_WT, mask_TC, mask_3], dtype=np.float32)


class HippocampusDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, volume_ids, input_dim=128, half_precision=False, device='cpu', rotate_volume=False):
        'Initialization'
        super(HippocampusDataset, self).__init__(volume_ids, input_dim=128, half_precision=False, device='cpu', rotate_volume=False)

    def load_X(self, volume_id, input_dim=128, rotate_by=None, on_axes=(-2,-1)):
        with gzip.GzipFile('data/HIPPO-3D/volumes/{}.npy.gz'.format(volume_id), 'r') as f:
            volume = np.load(f)
        
        if rotate_by:
            volume = interpolation.rotate(volume, axes=on_axes, angle=rotate_by)
        
        # Hippocampus
        original_dim = volume.shape[0]
        volume = interpolation.zoom(volume, zoom=input_dim/original_dim, order=0)
        return np.asarray([volume], dtype=np.float32)

    def load_y(self, volume_id, input_dim=128, rotate_by=None, on_axes=(-2,-1)):
        with gzip.GzipFile('data/HIPPO-3D/masks/{}.npy.gz'.format(volume_id), 'r') as f:
            mask = np.load(f)
        
        if rotate_by:
            mask = interpolation.rotate(mask, axes=on_axes, angle=rotate_by)
        
        original_dim = mask.shape[0]
        
        mask = interpolation.zoom(mask, zoom=input_dim/original_dim, order=0)
        
        mask_1 = (mask == 0.5).astype(int)
        mask_2 = (mask == 1).astype(int)
        
        return np.asarray([mask_1, mask_2], dtype=np.float32)