#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 12:13:12 2018

@author: Rongzhao Zhang
"""

from __future__ import print_function
import os
import os.path as P
import numpy as np
import scipy.misc
import torch.utils.data as udata

__all__ = ['Dataset_SEGCLS_png', 'MMDataset_memmap', ]


class General_Dataset_SEGCLS(udata.Dataset):
    """
    General Dataset
    INPUT:
        ROOT       -- Root directory of 'split' and 'datapath'
        SPLIT      -- Relative path of split file
                      e.g. 'split/train.txt'
        DATAPATH   -- Relative path of data folder
                      e.g. 'data/inbreast'
        MODALITIES -- Tuple of modality names, label must be the first one
                      e.g. ('label', 'img')
        TRANSFORM  -- Transform imposed on input data (default = None)
                      e.g. transform=my.transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(),])
        SHAPE      -- (For reading memmap data) Shape of memmap data
                      e.g. (512, 384)

    OUTPUT ( RETURNED BY __GETITEM__ ) :
        (IMG, LABEL) -- (C, H, W) as np.float32 and (H, W) as np.uint8 for ''img'' and ''label'' respectively
    """
    def __init__(self, root, split, datapath, modalities, transform=None, shape=None):
        self.transform = transform
        self.data = []
        self.label = []
        
        # Load subject names into a list
        split_fname = os.path.join(root, split)
        sn_list = open(split_fname, 'r').read().splitlines()
        
        for sn in sn_list:
            label_ = self.access_data(root, datapath, modalities[0], sn, 'uint8', shape)
            self.label.append(label_)
            
            img_ = []
            for mod in modalities[1:]:
                image = self.access_data(root, datapath, mod, sn, 'float32', shape)
                img_.append(image)
            img_ = np.stack(img_)
            self.data.append(img_)
            
    def access_data(self, root, datapath, mod, sn, dtype, shape=None):
        pass
    
    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]

        if self.transform is not None:
            img, label = self.transform(img, label)
            
        if label.sum() > 0:
            cls_label = 1
        else:
            cls_label = 0

        return img, label, cls_label
        
    def __len__(self):
        return len(self.data)


class Dataset_SEGCLS_png(General_Dataset_SEGCLS):
    def access_data(self, root, datapath, mod, sn, dtype, shape=None):
        fname = P.join(root, datapath, mod, '%s.png' % sn)
        data = scipy.misc.imread(fname)
        return data.astype(dtype)

class MMDataset_memmap(udata.Dataset):
    """
    Multi-Modality Dataset
    INPUT:
        ROOT       -- Root directory of 'split' and 'datapath'
        SPLIT      -- Relative path of split file
                      e.g. 'split/train.txt'
        DATAPATH   -- Relative path of data folder
                      e.g. 'data/'
        MODALITIES -- Tuple of modality names, label must be the first one
                      e.g. ('label', 'img')
        TRANSFORM  -- Transform imposed on input data (default = None)
                      e.g. transform=my.transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(),])

    OUTPUT ( RETURNED BY __GETITEM__ ) :
        (IMG, LABEL) -- (C, H, W) as np.float32 and (H, W) as np.uint8 for ''img'' and ''label'' respectively
    """

    def __init__(self, root, split, datapath, modalities, transform=None, shape=None):
        self.transform = transform
        self.data = []
        self.label = []

        # Load subject names into a list
        split_fname = os.path.join(root, split)
        sn_list = open(split_fname, 'r').read().splitlines()

        # Load images and labels
        for sn in sn_list:
            label_fn = os.path.join(root, datapath, modalities[0], '%s.dat' % sn)
            label_ = np.memmap(label_fn, dtype='uint8', mode='r', shape=shape[1:])

            img_fn = os.path.join(root, datapath, modalities[1], '%s.dat' % sn)
            img_ = np.memmap(img_fn, dtype='float32', mode='r', shape=shape)

            self.data.append(img_)
            self.label.append(label_)

#        self.data = np.stack(self.data) # (N, C, D, H, W)
#        self.label = np.stack(self.label) # (N, D, H, W)

#        del img_, label_

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]

        if self.transform is not None:
            img, label = self.transform(img, label)

        return img, label

    def __len__(self):
        return len(self.data)
        
        
        
        
        
        
        
        