from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class FER2013(data.Dataset):
 
    def __init__(self, split='Training', fold = 1, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.fold = fold # ignored
        if self.split == 'Training':
            self.data = h5py.File('FERTdata/fer2013_train.h5', 'r', driver='core')
        elif self.split == 'Testing':
            self.data = h5py.File('FERTdata/fer2013_test.h5', 'r', driver='core')
        
        # Load data
        self.train_data = self.data['data_pixel'][:]
        self.train_labels = self.data['data_label'][:]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.train_data)

