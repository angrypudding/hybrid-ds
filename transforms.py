#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:11:54 2017

@author: zrz
"""

from __future__ import division
import torch
import random
import numbers
import types
import numpy as np

__all__ = ['Compose', 'ToTensor', 'Normalize', 'Lambda', 'CenterCrop', 'RandomCrop',
           'RandomTransverseFlip', 'RandomSagittalFlip', 'RandomVerticalFlip',
           'RandomFlip', 'RandomDropout', 'RandomBlack', ]

def passthrough(img, label):
    return img, label

def has_even(intseq):
    for i in intseq:
        if i % 2 is 0:
            return True
    return False

def crop_size_correct(sp, ep, this_size):
    assert ep-sp <= this_size, 'Invalid crop size.'
    if sp < 0:
        ep -= sp
        sp -= sp
    elif ep > this_size:
        sp -= (ep-this_size)
        ep -= (ep-this_size)

    return sp, ep

def crop(tensor, locations):
    """ Crop on the inner-most 2 or 3 dimensions
    ''location'' is a tuple indicating locations of start and end points
    """
    s = tensor.size()
    if len(locations) == 6:
        x1, y1, z1, x2, y2, z2 = locations
        x1, x2 = crop_size_correct(x1, x2, s[-1])
        y1, y2 = crop_size_correct(y1, y2, s[-2])
        z1, z2 = crop_size_correct(z1, z2, s[-3])
        return tensor[..., z1:z2, y1:y2, x1:x2]
    elif len(locations) == 4:
        x1, y1, x2, y2 = locations
        x1, x2 = crop_size_correct(x1, x2, s[-1])
        y1, y2 = crop_size_correct(y1, y2, s[-2])
        return tensor[..., y1:y2, x1:x2]
    else:
        raise RuntimeError('Invalid crop size dimension.')

def center_crop(tensor, size):
    if len(size) == 3:
        d, h, w = tensor.size()[-3:]
        td, th, tw = size
        if d == td and w == tw and h == th:
            return tensor

        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        z1 = (d - td) // 2
        loc = (x1, y1, z1, x1 + tw, y1 + th, z1 + td)
        return crop(tensor, loc)
    elif len(size) == 2:
        h, w = tensor.size()[-2:]
        th, tw = size
        if w == tw and h == th:
            return tensor

        x1 = (w - tw) // 2
        y1 = (h - th) // 2
        loc = (x1, y1, x1 + tw, y1 + th)
        return crop(tensor, loc)
    else:
        raise RuntimeError('Invalid center crop size.')

def crop_centroid(tensor, centroid, size):
    """ Crop on the inner-most 2 or 3 dimensions
    ''centroid'' is a tuple indicating locations of the centroid
    ''size'' is a tuple indicating the size of the cropped patch
    """
    assert len(centroid) == len(size), 'Mismatched centroid and size.'
#    if has_even(size):
#        raise RuntimeWarning('Even crop size. May lead to one-voxel shift of the centroid.')

    s = [int(ss) // 2 for ss in size]
    start_pos = [ci-si for ci, si in zip(centroid, s)]
    end_pos = [start_pos_i + size_i for start_pos_i, size_i in zip(start_pos, size)]
    if len(centroid) == 3:
        locations = (start_pos[2], start_pos[1], start_pos[0], end_pos[2], end_pos[1], end_pos[0])
        return crop(tensor, locations)
    elif len(centroid == 2):
        locations = (start_pos[1], start_pos[0], end_pos[1], end_pos[0])
        return crop(tensor, locations)
    else:
        raise RuntimeError('Invalid centroid crop size.')


def flip_tensor(tensor, axis):
    if len(tensor.size()) == 1:
        return tensor
    tNp = np.flip(tensor.numpy(), axis).copy()
    return torch.from_numpy(tNp)


def rot90_tensor(tensor, k=1):
    if len(tensor.size()) == 2:
        tNp = np.rot90(tensor.numpy(), k).copy()
    elif len(tensor.size()) == 3:
        tNp = np.rot90(tensor.numpy(), k, (1,2)).copy()
    else:
        tNp = tensor.numpy()
    return torch.from_numpy(tNp)


class Compose(object):
    """ Composes several transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.RandomCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label


class ToTensor(object):
    """ Convert np.ndarray to torch.*Tensor """
    def __call__(self, img, label):
        return torch.from_numpy(img.copy()).float(), torch.from_numpy(label.copy()).long()

class Normalize(object):
    """ Normalize ''tensor'' by ''mean'' and ''std'' along each channel if corresponding arguments are provided.
        Other normalize to zero mean and unit std by channel.
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, label):
        if self.mean is None:
            self.mean = []
            for t in tensor:
                self.mean.append(t.mean())
        if self.std is None:
            self.std = []
            for t in tensor:
                self.std.append(t.std())

        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor, label

class Scale(object):
    """ TO BE IMPLEMENTED
    Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self):
        pass

    def __call__(self, img, label):
        return img, label

class Pad(object):
    """ Pad the input image to specified size with default 'pad_value' 0. """
    def __init__(self, size, pad_value=0):
        if len(size) == 3:
            self.padder = Pad3d(size, pad_value)
        elif len(size) == 2:
            self.padder = Pad2d(size, pad_value)
        else:
            raise RuntimeError('Invalid center crop size.')

    def __call__(self, img, label):
        img_new, label_new = self.padder(img, label)
        return img_new, label_new

class Pad3d(object):
    """ Pad the 3D input image to specified size with default 'pad_value' 0. """
    def __init__(self, size, pad_value=0):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, img, label):
        img_new = (torch.ones(img.size(0), *self.size) * self.pad_value).float()
        label_new = torch.zeros(*self.size).long()
        new_size = np.array(self.size)
        size = np.array(label.size())
        start_pos = (new_size - size) // 2
        end_pos = start_pos + size
#        print('new size is', new_size)
#        print('start_pos is', start_pos)
#        print('end_pos is', end_pos)
#        print('img_new size is', img_new.size())
#        print('img size is', img.size())
        img_new[..., start_pos[0]:end_pos[0], start_pos[1]:end_pos[1],
                start_pos[2]:end_pos[2]] = img
        label_new[..., start_pos[0]:end_pos[0], start_pos[1]:end_pos[1],
                start_pos[2]:end_pos[2]] = label
        return img_new, label_new

class Pad2d(object):
    """ Pad the 2D input image to specified size with default 'pad_value' 0. """
    def __init__(self, size, pad_value=0):
        self.size = size
        self.pad_value = pad_value

    def __call__(self, img, label):
        img_new = (torch.ones(img.size(0), *self.size) * self.pad_value).float()
        label_new = torch.zeros(*self.size).long()
        new_size = np.array(self.size)
        size = np.array(label.size())
        start_pos = (new_size - size) // 2
        end_pos = start_pos + size
        img_new[..., start_pos[0]:end_pos[0], start_pos[1]:end_pos[1]] = img
        label_new[..., start_pos[0]:end_pos[0], start_pos[1]:end_pos[1]] = label
        return img_new, label_new

class Lambda(object):
    """Applies a lambda as a transform"""
    def __init__(self, lambd):
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def __call__(self, img, label):
        return self.lambd(img, label)

class CenterCrop(object):
    def __init__(self, size, size_label):
        if len(size) == 3:
            self.cropper = CenterCrop3d(size, size_label)
        elif len(size) == 2:
            self.cropper = CenterCrop2d(size, size_label)
        else:
            raise RuntimeError('Invalid center crop size.')

    def __call__(self, img, label):
        return self.cropper(img, label)

class CenterCrop3d(object):
    def __init__(self, size, size_label=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size),) * 3
        else:
            self.size = size
        if size_label is None:
            self.size_label = self.size
        elif isinstance(size_label, numbers.Number):
            self.size_label = (int(size_label),) * 3
        else:
            self.size_label = size_label

    def __call__(self, img, label):
        return center_crop(img, self.size), \
               center_crop(label, self.size_label)

class CenterCrop2d(object):
    def __init__(self, size, size_label=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size),) * 2
        else:
            self.size = size
        if size_label is None:
            self.size_label = self.size
        elif isinstance(size_label, numbers.Number):
            self.size_label = (int(size_label),) * 2
        else:
            self.size_label = size_label

    def __call__(self, img, label):
        return center_crop(img, self.size), \
               center_crop(label, self.size_label)

class RandomCrop(object):
    """Crops the given (img, label) at a random location to have a region of
    the given size. size MUST be a 3-tuple (target_depth, target_height, target_width)
    or a 2-tuple (target_height, target_width). The dimensionality is automatically
    detected according to the length of the size tuple.
    """
    def __init__(self, size):
        if len(size) == 3:
            self.cropper = RandomCrop3d(size)
        elif len(size) == 2:
            self.cropper = RandomCrop2d(size)
        else:
            raise RuntimeError('Invalid random crop size.')

    def __call__(self, img, label):
        return self.cropper(img, label)

class RandomCropMinSize(object):
    """Crops the given (img, label) at a random location to be of the given size, 
    while ensuring the number of positive pixels is either zero or larger than 
    a minimal value. Size MUST be a 3-tuple (target_depth, target_height, target_width)
    or a 2-tuple (target_height, target_width). The dimensionality is automatically
    detected according to the length of the size tuple.
    """
    def __init__(self, size, mini_positive=0):
        self.mini_positive = mini_positive
        if len(size) == 3:
            self.cropper = RandomCrop3d(size)
        elif len(size) == 2:
            self.cropper = RandomCrop2d(size)
        else:
            raise RuntimeError('Invalid random crop size.')

    def __call__(self, img, label):
        imgc, labelc = self.cropper(img, label)
        count = 0
        while(0 < labelc.sum() < self.mini_positive):
            imgc, labelc = self.cropper(img, label)
            count += 1
        if count > 0:
            print('Crop %d times for a valid positive size.' % count)
        return imgc, labelc

class RandomCrop3d(object):
    """Crops the given (img, label) at a random location to have a region of
    the given size. size can be a tuple (target_depth, target_height, target_width)
    or an integer, in which case the target will be of a cubic shape (size, size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        d, h, w = img.size()[-3:]
        td, th, tw = self.size
        if d == td and w == tw and h == th:
            return img, label

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        z1 = random.randint(0, d - td)
        loc = (x1, y1, z1, x1 + tw, y1 + th, z1 + td)
        return crop(img, loc), crop(label, loc)

class RandomCrop2d(object):
    """Crops the given (img, label) at a random location to have a region of
    the given size. size can be a tuple (target_depth, target_height, target_width)
    or an integer, in which case the target will be of a cubic shape (size, size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        h, w = img.size()[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return img, label

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        loc = (x1, y1, x1 + tw, y1 + th)
        return crop(img, loc), crop(label, loc)

class BalanceCrop(object):
    """Randomly crop the given image and label with balanced centroid class.
    """
    def __init__(self, prob, img_size, label_size=None):
        self.prob = prob
        if label_size is None:
            label_size = img_size
        if isinstance(img_size, numbers.Number):
            img_size = (int(img_size),) * 3
        if isinstance(label_size, numbers.Number):
            label_size = (int(label_size),) * 3

        self.img_size = torch.LongTensor(img_size)
        self.label_size = torch.LongTensor(label_size)
        self.mask = None

    def __call__(self, img, label):
        # Note: img is CxDxHxW, label is DxHxW
#        if self.mask is None or self.mask.size() != img.size():
#            self._make_mask(label.size())

        positive_loc = torch.nonzero(label) #*self.mask

        inverse_label = torch.eq(label, 0).long()
        negative_loc = torch.nonzero(inverse_label) #*self.mask

        if len(negative_loc) == 0 and len(positive_loc) == 0:
            raise RuntimeError('Invalid patch size.')
        elif len(negative_loc) == 0:
            is_positive = True
        elif len(positive_loc) == 0:
            is_positive = False
        else:
            is_positive = random.random() <= self.prob

        if is_positive:
            i = random.randint(0, len(positive_loc)-1)
            center_loc = positive_loc[i]
        else:
            i = random.randint(0, len(negative_loc)-1)
            center_loc = negative_loc[i]

        return crop_centroid(img, center_loc, self.img_size), \
               crop_centroid(label, center_loc, self.label_size)

    def _make_mask(self, input_size):
        mask = torch.ones(input_size).long()
        half_img_size = self.img_size / 2
        if len(input_size) == 3:
            mask[:half_img_size[0]] = 0
            mask[-half_img_size[0]:] = 0
            mask[:,:half_img_size[1]] = 0
            mask[:,-half_img_size[1]:] = 0
            mask[:,:,:half_img_size[2]] = 0
            mask[:,:,-half_img_size[2]:] = 0
        elif len(input_size) == 2:
            mask[:half_img_size[0]] = 0
            mask[-half_img_size[0]:] = 0
            mask[:,:half_img_size[1]] = 0
            mask[:,-half_img_size[1]:] = 0
        else:
            raise RuntimeError('Unknown label dimension.')
        self.mask = mask

class RandomTransverseFlip(object):
    """ Randomly transverse flips the given Tensor along 'w' dimension
    """
    def __call__(self, img, label):
        return flip_tensor(img, -1), flip_tensor(label, -1)

class RandomSagittalFlip(object):
    """ Randomly sagittally flips the given Tensor along 'h' dimension
    """
    def __call__(self, img, label):
        return flip_tensor(img, -2), flip_tensor(label, -2)

class RandomVerticalFlip(object):
    """ Randomly vertically flips the given Tensor along 'd' dimension
    """
    def __call__(self, img, label):
        return flip_tensor(img, -3), flip_tensor(label, -3)

class RandomFlip(object):
    def __init__(self, axis_switch=None):
        if len(axis_switch) == 3:
            self.flipper = RandomFlip3d(axis_switch)
        elif len(axis_switch) == 2:
            self.flipper = RandomFlip2d(axis_switch)
        elif axis_switch == None:
            self.flipper = passthrough
        else:
            raise RuntimeError('Invalid random flip controller.')

    def __call__(self, img, label):
        return self.flipper(img, label)

class RandomFlip3d(object):
    def __init__(self, axis_switch=(1,1,1)):
        self.axis_switch = axis_switch

    def __call__(self, img, label):
        if self.axis_switch[0]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -3)
                label = flip_tensor(label, -3)
        if self.axis_switch[1]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -2)
                label = flip_tensor(label, -2)
        if self.axis_switch[2]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -1)
                label = flip_tensor(label, -1)
        return img, label

class RandomFlip2d(object):
    def __init__(self, axis_switch=(1,1)):
        self.axis_switch = axis_switch

    def __call__(self, img, label):
        if self.axis_switch[0]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -2)
                label = flip_tensor(label, -2)
        if self.axis_switch[1]:
            if random.randint(0,1) == 1:
                img = flip_tensor(img, -1)
                label = flip_tensor(label, -1)
        return img, label

class RandomSizedCrop(object):
    """ TO BE IMPLEMENTED
    Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self):
        pass

    def __call__(self, img, label):
        return img, label


class RandomRotate2d(object):
    def __init__(self):
        pass

    def __call__(self, img, label):
        k = random.randint(0, 3)
        if k == 0:
            return img, label
        return rot90_tensor(img, k), rot90_tensor(label, k)


class RandomDropout(object):
    """ Randomly drop an input channel / modality """
    def __init__(self, drop_rate=0.5):
        self.drop_rate = drop_rate

    def __call__(self, tensor, label):
        if self.drop_rate <= 0:
            return tensor, label
        elif self.drop_rate > 1:
            raise RuntimeError('Dropout rate greater than 1.')

        C = tensor.size(0)
        drop_count = 0
        rand_flag = np.random.random(C)
        rand_flag = rand_flag < self.drop_rate
        if all(rand_flag):
            rand_flag[random.randint(0, C-1)] = False
        for c in range(C):
            if rand_flag[c]:
                drop_count += 1
                tensor[c, ...] = 0.0
#        print(rand_flag)
#        print(drop_count)
        tensor *= C / (C-drop_count)
        return tensor, label


class RandomBlack(object):
    """ Randomly set patches to zeor"""
    def __init__(self, black_patch_size=None):
        self.black_patch_size = black_patch_size
        if black_patch_size is None:
            self.blacker = passthrough
        elif len(black_patch_size) == 2:
            self.blacker = RandomBlack2d(black_patch_size)
        elif len(black_patch_size) == 3:
            self.blacker = RandomBlack3d(black_patch_size)
        else:
            raise RuntimeError('Invalid length of black_patch_size.')
        
    def __call__(self, tensor, label):
        return self.blacker(tensor, label)
        

class RandomBlack2d(object):
    """ Randomly set patches to zeor"""
    def __init__(self, black_patch_size=None):
        self.black_patch_size = black_patch_size
        
    def __call__(self, tensor, label):
        th, tw = self.black_patch_size
        h, w = tensor.size()[-2:]
        x1 = random.randint(0, h-th)
        y1 = random.randint(0, w-tw)
        tensor[..., x1:x1+th, y1:y1+tw] = 0
        label[..., x1:x1+th, y1:y1+tw] = 0
        
        return tensor, label


class RandomBlack3d(object):
    """ Randomly set patches to zeor"""
    def __init__(self, black_patch_size=None):
        self.black_patch_size = black_patch_size
        
    def __call__(self, tensor, label):
        td, th, tw = self.black_patch_size
        d, h, w = tensor.size()[-3:]
        x1 = random.randint(0, d-td)
        y1 = random.randint(0, h-th)
        z1 = random.randint(0, w-tw)
        tensor[..., x1:x1+td, y1:y1+th, z1:z1+tw] = 0
        label[..., x1:x1+td, y1:y1+th, z1:z1+tw] = 0
        
        return tensor, label
    

##### Backups #####
class RandomCrop_bak(object):
    """Crops the given (img, label) at a random location to have a region of
    the given size. size can be a tuple (target_depth, target_height, target_width)
    or an integer, in which case the target will be of a cubic shape (size, size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, label):
        d, h, w = img.size()[-3:]
        td, th, tw = self.size
        if d == td and w == tw and h == th:
            return img, label

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        z1 = random.randint(0, d - td)
        loc = (x1, y1, z1, x1 + tw, y1 + th, z1 + td)
        return crop(img, loc), crop(label, loc)

