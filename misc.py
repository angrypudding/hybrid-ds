#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:52:38 2017

@author: zhang
"""
import time
import numpy as np
import os.path as osp
import math
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn
import transforms
import general_dataset as datasets

__all__ = ['timestr', 'bilinear_init2d', 'weights_init', 'DataHub']

def timestr(form=None):
    if form is None:
        return time.strftime("<%Y-%m-%d %H:%M:%S>", time.localtime())
    if form == 'mdhm':
        return time.strftime('%m%d%H%M', time.localtime())

def bilinear_init3d(m, method=nn.init.kaiming_normal):
    inC, outC, d, h, w = m.weight.size()
    if not (outC == 1 and inC == m.groups and h==w==4):
        method(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    else:
        kernel = torch.zeros(d, h, w)
        f_d = math.ceil(d / 2.)
        f_h = math.ceil(h / 2.)
        f_w = math.ceil(w / 2.)
        c_d = (2 * f_d - 1 - f_d % 2) / (2. * f_d)
        c_h = (2 * f_h - 1 - f_h % 2) / (2. * f_h)
        c_w = (2 * f_w - 1 - f_w % 2) / (2. * f_w)
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    kernel[i,j,k] = (1-math.fabs(i/f_d-c_d)) * \
                          (1-math.fabs(j/f_h-c_h)) * (1-math.fabs(k/f_w-c_w))
        for i in range(inC):
            m.weight.data[i] = kernel

def bilinear_init2d(m, method=nn.init.kaiming_normal):
    inC, outC, h, w = m.weight.size()
    if not (outC == 1 and inC == m.groups and h==w==4):
        method(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    else:
        kernel = torch.zeros(h, w)
        f_h = math.ceil(h / 2.)
        f_w = math.ceil(w / 2.)
        c_h = (2 * f_h - 1 - f_h % 2) / (2. * f_h)
        c_w = (2 * f_w - 1 - f_w % 2) / (2. * f_w)
        for j in range(h):
            for k in range(w):
                kernel[j, k] = (1-math.fabs(j/f_h-c_h)) * (1-math.fabs(k/f_w-c_w))
        for i in range(inC):
            m.weight.data[i] = kernel

def weights_init(m, method=nn.init.kaiming_normal):
    '''
    ConvXd: kaiming_normal (weight), zeros (bias)
    BatchNormXd: ones (weight), zeros (bias)
    ConvTransposedXd: bilinear (weight), no bias
    '''
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1 or classname.find('Conv2d') != -1:
        method(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm3d') != -1 or classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
    elif classname.find('ConvTranspose3d') != -1:
        bilinear_init3d(m, method)
    elif classname.find('ConvTranspose2d') != -1:
        bilinear_init2d(m, method)

class DataHub(object):
    def __init__(self, root, train_split, val_split, test_split, datapath, train_batchsize,
                 test_batchsize, modalities, std=1, mean=0, modal_test=None, datapath_test=None, 
                 rand_flip=None, crop_type=None, crop_size_img=None, crop_size_label=None, 
                 balance_rate=0.5, train_pad_size=None, test_pad_size=None, mod_drop_rate=0, 
                 train_drop_last=False, crop_type_test=None, crop_size_img_test=None, 
                 crop_size_label_test=None, DataSet=datasets.Dataset_SEGCLS_png, TestDataSet=None, 
                 label_loader_path=None, weighted_sample_rate=None, rand_rot90=False, 
                 num_workers=1, mem_shape=None, random_black_patch_size=None, 
                 mini_positive=None):
        self.root = root
        self.std = std
        self.mean = mean
        self.num_workers = num_workers
        self.mem_shape = mem_shape
        if TestDataSet is None:
            TestDataSet = DataSet
        if datapath_test is None:
            datapath_test = datapath
        if modal_test is None:
            modal_test = modalities

        if train_split:
            with open(osp.join(root, train_split), 'r') as f:
                self._train_sn = f.read().splitlines()
        if val_split:
            with open(osp.join(root, val_split), 'r') as f:
                self._val_sn = f.read().splitlines()
        if test_split:
            with open(osp.join(root, test_split), 'r') as f:
                self._test_sn = f.read().splitlines()
            
        if osp.exists(osp.join(root, datapath, 'meanstd.txt')):
            with open(osp.join(root, datapath, 'meanstd.txt'), 'r') as f:
                lines = f.read().splitlines()
            self.mean = [ float(x) for x in lines[0].split()[1:] ]
            self.std = [ float(x) for x in lines[1].split()[1:] ]
            print('import mean and std value from file \'meanstd.txt\'')
            print('mean = %s, std = %s' % (str(self.mean), str(self.std)))

        self.basic_transform_ops = [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]

        train_transform = \
        self._make_train_transform(crop_type, crop_size_img, crop_size_label,
                                   rand_flip, mod_drop_rate, balance_rate,
                                   train_pad_size, rand_rot90, random_black_patch_size,
                                   mini_positive)
        test_transform = \
        self._make_test_transform(crop_type_test, crop_size_img_test,
                                  crop_size_label_test, test_pad_size)

        self._trainloader = \
        self._make_dataloader(DataSet, train_split, datapath, train_batchsize, train_transform,
                              modalities, shuffle=True, drop_last=train_drop_last,
                              weighted_sample_rate=weighted_sample_rate)

        self._trainseqloader = \
        self._make_dataloader(TestDataSet, train_split, datapath_test, test_batchsize,
                              test_transform, modal_test, shuffle=False)

        self._valloader = \
        self._make_dataloader(TestDataSet, val_split, datapath_test, test_batchsize,
                              test_transform, modal_test, shuffle=False)

        self._testloader = \
        self._make_dataloader(TestDataSet, test_split, datapath_test, test_batchsize,
                              test_transform, modal_test, shuffle=False)

        self._trainseqloader_label = \
        self._make_labelloader(train_split, label_loader_path, test_batchsize)

        self._valloader_label = \
        self._make_labelloader(val_split, label_loader_path, test_batchsize)

        self._testloader_label = \
        self._make_labelloader(test_split, label_loader_path, test_batchsize)


    def _make_dataloader(self, DataSet, split, datapath, batch_size, transform, modalities,
                         shuffle=False,drop_last=False, weighted_sample_rate=None):
        if split is None:
            return None
        if (self.mem_shape is not None) and DataSet == datasets.MMDataset_memmap:
            data_set = DataSet(self.root, split, datapath, modalities, transform, self.mem_shape)
        else:
            data_set = DataSet(self.root, split, datapath, modalities, transform)
        sampler = None
        if weighted_sample_rate is not None:
            weights = np.where(data_set.get_mask() == 1, weighted_sample_rate[1],
                               weighted_sample_rate[0]).tolist()
#            weights = torch.from_numpy(weights).float()
            sampler = WeightedRandomSampler(weights, len(data_set), replacement=True)
            shuffle = False
        data_loader = DataLoader(data_set, batch_size=batch_size, sampler=sampler,
                                 shuffle=shuffle, num_workers=self.num_workers, pin_memory=False,
                                 drop_last=drop_last)
        return data_loader

    def _make_labelloader(self, split, datapath, batch_size):
        if datapath is None:
            return None
        data_set = datasets.VanillaDataset(self.root, split, datapath)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=False, drop_last=False)
        return data_loader

    def _make_train_transform(self, crop_type, crop_size_img, crop_size_label,
                             rand_flip, mod_drop_rate, balance_rate, pad_size,
                             rand_rot90, random_black_patch_size, mini_positive):
        train_transform_ops = self.basic_transform_ops.copy()
            
        train_transform_ops += [transforms.RandomBlack(random_black_patch_size),
                                transforms.RandomDropout(mod_drop_rate),
                                transforms.RandomFlip(rand_flip)]
        if pad_size is not None:
            train_transform_ops.append(transforms.Pad(pad_size, 0))

        if rand_rot90:
            train_transform_ops.append(transforms.RandomRotate2d())

        if crop_type == 'random':
            if mini_positive:
                train_transform_ops.append(transforms.RandomCropMinSize(crop_size_img, mini_positive))
            else:
                train_transform_ops.append(transforms.RandomCrop(crop_size_img))
        elif crop_type == 'balance':
            train_transform_ops.append(transforms.BalanceCrop(balance_rate, crop_size_img,
                                                              crop_size_label))
        elif crop_type == 'center':
            train_transform_ops.append(transforms.CenterCrop(crop_size_img,
                                                          crop_size_label))
        elif crop_type is None:
            pass
        else:
            raise RuntimeError('Unknown train crop type.')

        return transforms.Compose(train_transform_ops)

    def _make_test_transform(self, crop_type, crop_size_img, crop_size_label, pad_size):
        test_transform_ops = self.basic_transform_ops.copy()
        if pad_size is not None:
            test_transform_ops.append(transforms.Pad(pad_size, 0))
        if crop_type == 'center':
            test_transform_ops.append(transforms.CenterCrop(crop_size_img,
                                                          crop_size_label))
        elif crop_type is None:
            pass
        else:
            raise RuntimeError('Unknown test crop type.')

        return transforms.Compose(test_transform_ops)

    def trainloader(self):
        return self._trainloader

    def valloader(self):
        return self._valloader

    def testloader(self):
        return self._testloader

    def trainseqloader(self):
        return self._trainseqloader

    def valloader_label(self):
        return self._valloader_label

    def testloader_label(self):
        return self._testloader_label

    def trainseqloader_label(self):
        return self._trainseqloader_label

    def train_sn(self):
        return self._train_sn

    def val_sn(self):
        return self._val_sn

    def test_sn(self):
        return self._test_sn


