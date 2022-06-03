import random
import torch
import cv2
import numpy as np
import os

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, labels, dataset_type='train', input_dim=128, half_precision=False, device='cpu', random_seed=9, data_dir=''):
        'Initialization'

        self.input_dim=input_dim
        self.half_precision = half_precision
        self.dataset_type = dataset_type
        self.data_dir=data_dir
        self.id_label_dict = self.load_label()

        self.image_ids = []
        
        self.labels = labels

    def __len__(self):
        return len(self.image_ids)

    def load_label(self):
        with open('{}/{}-labels.txt'.format(self.data_dir, self.dataset_type), 'r') as f:
            text = f.readlines()
            id_label_dict = {line.split()[0]: line.split()[1] for line in text}
        return id_label_dict

    def load_X(self):
        pass

    def load_y(self):
        pass

    def set_input_dim(self, randomize=True, permute_from=[], input_dim=None):
        if randomize:
            self.input_dim = random.choice(permute_from)
        else:
            self.input_dim = input_dim

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image_id = self.image_ids[index]
        X, y = self.load_X(image_id, input_dim=self.input_dim), self.load_y(image_id)
        return X, y

class ROCTDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, labels, dataset_type='train', input_dim=128, half_precision=False, device='cpu', random_seed=9, data_dir=''):
        super(ROCTDataset, self).__init__(labels, dataset_type, input_dim, half_precision, device, random_seed, data_dir)
        
        for i in range(len(labels)):
            self.image_ids.append([k for k, v in self.id_label_dict.items() if v == str(i)])
        
        max_count = min(len(l) for l in self.image_ids)
        
        self.image_ids = [id for ids in self.image_ids for id in random.Random(random_seed).sample(ids, max_count)]
        random.Random(random_seed).shuffle(self.image_ids)

    def load_X(self, image_id, input_dim=128):
        label = self.labels[int(self.id_label_dict[image_id])]
        image = cv2.imread('{}/{}/{}/{}'.format(self.data_dir,self.dataset_type, label, image_id), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (input_dim, input_dim))

        image = image/255

        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        return image

    def load_y(self, image_id):
        return int(self.id_label_dict[image_id])