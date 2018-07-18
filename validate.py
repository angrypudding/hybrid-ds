#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 14:47:07 2018

@author: rongzhao
"""

import torch
import torch.nn.functional as F
import os
import os.path as P
import numpy as np
from scipy.misc import toimage
from scipy import ndimage
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score

def is_float(value):
    if isinstance(value, torch.Tensor):
        return value.is_floating_point()
    return isinstance(value, float)

'''All following metrics are calculated as Tensor'''
def dice(pred_b, target_b):
    '''Calculate Dice index of the binary prediction'''
    eps = 1e-6
    dice_index = (2*(pred_b*target_b).sum().float() + eps) / (pred_b.sum().float() + target_b.sum().float() + eps)
    return dice_index

def accuracy(pred_b, target_b):
    '''Calculate accuracy of the binary prediction'''
    accuracy = ((pred_b==target_b).sum().float()) / torch.tensor(target_b.numel(), dtype=torch.float, device=target_b.device)
    return accuracy

def sensitivity(pred_b, target_b):
    '''Calculate sensitivity of the binary prediction'''
    eps = 1e-6
    sensitivity_index = ((pred_b*target_b).sum().float() + eps) / (target_b.sum().float() + eps)
    return sensitivity_index

def specificity(pred_b, target_b):
    '''Calculate specificity of the binary prediction'''
    device = pred_b.device
    zero, one = torch.tensor(0, device=device), torch.tensor(1, device=device)
    eps = 1e-6
    pred_bn = torch.where(pred_b>0, zero, one)
    target_bn = torch.where(target_b>0, zero, one)
    specificity_index = ((pred_bn*target_bn).sum().float() + eps) / (target_bn.sum().float() + eps)
    return specificity_index

def sizeL(pred_b, target_b):
    return target_b.sum().float()

def sizeP(pred_b, target_b):
    return pred_b.sum().float()

def precision(pred_b, target_b):
    '''Calculate precision of the binary prediction'''
    eps = 1e-6
    prec = ((pred_b*target_b).sum().float() + eps) / (pred_b.sum().float() + eps)
    return prec

def auc(prob, target_b):
    '''Calculate area under ROC curve (AUC) score'''
    prob_np = prob.cpu().numpy()
    target_np = target_b.cpu().numpy()
    try:
        return roc_auc_score(target_np, prob_np)
    except ValueError: # In case only one class present
        return 1

def num_component(mask):
    mask_np = mask.cpu().numpy()
    _, num_compo = ndimage.label(mask_np, np.ones((3,3)))
    num_compo = torch.tensor(num_compo, dtype=torch.float32, device=mask.device)
    return num_compo
    
def num_false_positive(pred_b, target_b):
    pred_np = pred_b.cpu().numpy()
    target_np = target_b.cpu().numpy()
    false_counter = 0
    pred_compo, num_compo = ndimage.label(pred_np, np.ones((3,3)))
    for i in range(1, num_compo+1):
        current_pred = np.where(pred_compo == i, 1, 0)
        overlap = target_np * current_pred
        if ~overlap.any():
            false_counter += 1
    return torch.tensor(false_counter, dtype=torch.float32, device=target_b.device)

def num_positive(pred_b, target_b):
    return num_component(target_b)

def num_false_negative(pred_b, target_b):
    return num_false_positive(target_b, pred_b)

def num_true_positive(pred_b, target_b):
    return num_positive(pred_b, target_b) - num_false_negative(pred_b, target_b)

def prob2label(pred):
    '''Retrieve image-wise label from the raw network output (can be of (C,) for CNN, 
    or (C,H,W) for multi-instance learning CNN)'''
    if len(pred.size()) == 1:
        _, label = torch.max(pred, dim=0)
        return label
    pred = pred.view(pred.size(0), -1)
    mini_nega_loc = torch.argmin(pred[0])
    label = torch.argmax(pred[:, mini_nega_loc])
    return label

def raw2prob(pred):
    '''Retrieve image-wise label probability from the raw network output (can be of (C,) for CNN, 
    or (C,H,W) for multi-instance learning CNN)'''
    if len(pred.size()) == 1:
        prob = F.softmax(pred, dim=0)[1]
        return prob
    pred = pred.view(pred.size(0), -1)
    mini_nega_loc = torch.argmin(pred[0])
    prob = F.softmax(pred[:, mini_nega_loc], dim=0)[1]
    return prob
    

class SegMetric(object):
    def __init__(self, sn_list=None, metric_names=None, is_cc=False):
        self.ACC = 'acc'
        self.DSC = 'dsc'
        self.SENS = 'sens'
        self.SPEC = 'spec'
        self.SIZEL = 'sizeL'
        self.SIZEP = 'sizeP'
        self.FPL = 'fpl'
        self.FNL = 'fnl'
        self.TOTALL = 'totall'
        self.ALL_METRIC = (self.ACC, self.DSC, self.SENS, self.SPEC, self.SIZEL, self.SIZEP)
        self.is_cc = is_cc
        if is_cc:
            self.ALL_METRIC = self.ALL_METRIC + (self.FPL, self.FNL, self.TOTALL)
        self.calculator = {
                self.ACC: accuracy,
                self.DSC: dice,
                self.SENS: sensitivity,
                self.SPEC: specificity,
                self.SIZEL: sizeL,
                self.SIZEP: sizeP,
                self.FPL: num_false_positive,
                self.FNL: num_false_negative,
                self.TOTALL: num_positive,
                }
        if metric_names == None:
            self.metric_names = self.ALL_METRIC
        else:
            for m in metric_names:
                if m not in self.ALL_METRIC:
                    raise RuntimeError('Unknown specified metric type: %s' % m)
            self.metric_names = metric_names
        
        self.sn_list = sn_list if sn_list else []
        self.buffer = dict()
        self.metric = dict()
        for m in self.metric_names:
            self.buffer[m] = []
            self.metric[m] = 0
        self.buffer_changed = True
        
    def _buffer2metric(self, is_strict=True):
        if self.buffer_changed:
            if len(self) == 0:
                self.buffer_changed = False
                return
            if is_strict:
                assert len(self.sn_list) == len(self.buffer[self.metric_names[0]]), 'Unmatch: lengths of sn_list and buffer.'
            for m in self.metric_names:
                self.metric[m] = float(torch.stack(self.buffer[m]).mean())
        self.buffer_changed = False
        return self.metric
    
    def get_metric(self):
        return self._buffer2metric()
    
    def write_metric(self, fid, preline=None, is_indiv=False):
        '''Write (final) detailed metrics to file'''
        self._buffer2metric()
        if preline:
            fid.write(preline + '\n')
        # Construct total_line and title_line
        metric_str = []
        title_line = '|%12s|' % 'SN'
        for k, v in self.metric.items():
            title_line += '%8s|' % str.upper(k)
            if is_float(v):
                s = '%s = %.4f' % (k, v)
            else:
                s = '%s = %d' % (k, v)
            metric_str.append(s)
        total_line = ', '.join(metric_str)
        fid.write(total_line + '\n')
        if is_indiv:
            fid.write(title_line + '\n')
            # Construct individual lines
            for i, sn in enumerate(self.sn_list):
                line = '|%12s|' % sn
                for _, v in self.buffer.items():
                    w = v[i]
                    if is_float(w):
                        s = '%8.4f|' % w
                    else:
                        s = '%8d|' % w
                    line += s
                fid.write(line + '\n')
        
    def print_metric(self, preword=None, is_indiv=False):
        '''Print (final, detailed) metrics to stdout'''
        self._buffer2metric()
        if preword:
            print('%s Segmentation Metrics:' % preword)
        else:
            print('Segmentation Metrics:')
        # Construct total_line and title_line
        metric_str = []
        title_line = '|%12s|' % 'SN'
        for k, v in self.metric.items():
            title_line += '%8s|' % str.upper(k)
            if is_float(v):
                s = '%s = %.4f' % (k, v)
            else:
                s = '%s = %d' % (k, v)
            metric_str.append(s)
        total_line = ', '.join(metric_str)
        print(total_line)
        if is_indiv:
            print(title_line)
            # Construct individual lines
            for i, sn in enumerate(self.sn_list):
                line = '|%12s|' % sn
                for _, v in self.buffer.items():
                    w = v[i]
                    if is_float(w):
                        s = '%8.4f|' % w
                    else:
                        s = '%8d|' % w
                    line += s
                print(line)

    def evaluate_append(self, pred, label, sn=None):
        self.buffer_changed = True
        if sn is not None:
            if len(self.sn_list) != len(self.buffer[self.metric_names[0]]):
                raise RuntimeWarning('SN is specified but the lengths of sn_list and buffer do not match.')
            else:
                self.sn_list.append(sn)
        for m in self.metric_names:
            self.buffer[m].append(self.calculator[m](pred, label))
        
    def append(self, sn, metrics):
        self.buffer_changed = True
        self.sn_list.append(sn)
        for i, m in enumerate(metrics):
            self.buffer[self.ALL_METRIC[i]].append(m)
            
    def __len__(self):
        return len(self.buffer[self.metric_names[0]])


class ClsMetric(object):
    def __init__(self, sn_list=None, metric_names=None):
        self.ACC = 'acc'
        self.F1 = 'f1'
        self.RECL = 'recl'
        self.PREC = 'prec'
        self.AUC = 'auc'
        self.ALL_METRIC = (self.ACC, self.F1, self.RECL, self.PREC, self.AUC)
        self.calculator = {
                self.ACC: accuracy,
                self.F1: dice,
                self.RECL: sensitivity,
                self.PREC: precision,
                self.AUC: auc}
        if metric_names == None:
            self.metric_names = self.ALL_METRIC
        else:
            for m in metric_names:
                if m not in self.ALL_METRIC:
                    raise RuntimeError('Unknown specified metric type: %s' % m)
            self.metric_names = metric_names
        
        self.sn_list = sn_list if sn_list else []
        self.buffer = dict(pred=[], prob=[], target=[])
        self.metric = dict()
        for m in self.metric_names:
            self.metric[m] = 0
        self.buffer_changed = True
            
    def _buffer2metric(self):
        if self.buffer_changed:
            prob_tensor = torch.stack(self.buffer['prob'])
            pred_tensor = torch.stack(self.buffer['pred'])
            target_tensor = torch.stack(self.buffer['target'])
            for m in self.metric_names:
                if m == self.AUC:
                    self.metric[m] = float(self.calculator[m](prob_tensor, target_tensor))
                else:
                    self.metric[m] = float(self.calculator[m](pred_tensor, target_tensor))
        self.buffer_changed = False
        return self.metric
    
    def get_metric(self):
        return self._buffer2metric()
    
    def write_metric(self, fid, preline=None, is_indiv=False):
        '''Write (final) detailed metrics to file'''
        self._buffer2metric()
        if preline:
            fid.write(preline + '\n')
        # Construct total_line and title_line
        metric_str = []
        for k, v in self.metric.items():
            if is_float(v):
                s = '%s = %.4f' % (k, v)
            else:
                s = '%s = %d' % (k, v)
            metric_str.append(s)
        total_line = ', '.join(metric_str)
        title_line = '|%12s|%8s|%8s|' % ('SN', 'Predict', 'Truth')
        fid.write(total_line + '\n')
        if is_indiv:
            fid.write(title_line + '\n')
            # Construct individual lines
            for i, sn in enumerate(self.sn_list):
                line = '|%12s|' % sn
                for _, v in self.buffer.items():
                    w = v[i]
                    s = '%8d|' % w
                    line += s
                fid.write(line + '\n')
        
    def print_metric(self, preword=None, is_indiv=False):
        '''Print (final, detailed) metrics to stdout'''
        self._buffer2metric()
        if preword:
            print('%s Classification Metrics:' % preword)
        else:
            print('Classification Metrics:')
        # Construct total_line and title_line
        metric_str = []
        title_line = '|%12s|%8s|%8s|' % ('SN', 'Predict', 'Truth')
        for k, v in self.metric.items():
            if is_float(v):
                s = '%s = %.4f' % (k, v)
            else:
                s = '%s = %d' % (k, v)
            metric_str.append(s)
        total_line = ', '.join(metric_str)
        print(total_line)
        if is_indiv:
            print(title_line)
            # Construct individual lines
            for i, sn in enumerate(self.sn_list):
                line = '|%12s|' % sn
                for _, v in self.buffer.items():
                    w = v[i]
                    s = '%8d|' % w
                    line += s
                print(line)

    def evaluate_append(self, prob, label, sn=None, thres=0.5):
        self.buffer_changed = True
        if sn is not None:
            if len(self.sn_list) != len(self.buffer['pred']):
                raise RuntimeWarning('SN is specified but the lengths of sn_list and buffer do not match.')
            else:
                self.sn_list.append(sn)
        self.buffer['prob'].append(prob)
        self.buffer['pred'].append((prob>thres).long())
        self.buffer['target'].append(label)
    
    def append(self, sn, prob, label):
        self.evaluate_append(prob, label, sn)

    def __len__(self):
        return len(self.buffer['target'])


def validate_cs(model, dataloader, sn_list, device, num_mo=1, save_dir=None, seg_thres=0.5, 
                cls_thres=0.5, is_cc=False, clsfromseg=False):
    '''Validate multi-ouput model's seg and cls performance on specified dataloader
    
    Arguements:
        Output: a SegMetric array and a ClsMetric array, elements in each array 
        correspond to multiple model outputs.
    '''
    sm, cm, sn_counter = [], [], []
    mo_ind = list(range(-num_mo, 0))
    for _ in mo_ind:
        sm.append(SegMetric(sn_list, is_cc=is_cc))
        cm.append(ClsMetric(sn_list))
        sn_counter.append(-1)
    if save_dir:
        seg_dir, cls_dir = P.join(save_dir, 'seg'), P.join(save_dir, 'cls')
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(cls_dir, exist_ok=True)
    with torch.no_grad():
        model.to(device)
        model.eval()
        for images, masks, labels in iter(dataloader):
            images = images.to(device)
            masks, labels = masks.to(device), labels.to(device)
            preds_seg, preds_cls = model(images)
            for i in mo_ind: # for each of the multiple outputs (heads)
                for j in range(len(preds_seg[i])): # within each mini-batch
                    idx = sn_counter[i] = sn_counter[i] + 1
                    seg_out = preds_seg[i, j]
#                    _, seg_mask = torch.max(seg_out, dim=0)
                    seg_prob = F.softmax(seg_out, dim=0)
                    seg_mask = (seg_prob[1] > seg_thres).long()
                    sm[i].evaluate_append(seg_mask, masks[j])
                    cls_out = preds_cls[i, j]
                    cls_prob = raw2prob(cls_out)
                    if clsfromseg:
                        cm[i].evaluate_append(seg_prob[1].max(), labels[j], thres=cls_thres)
                    else:
                        cm[i].evaluate_append(cls_prob, labels[j], thres=cls_thres)
                    if save_dir:
                        if i == -1:
                            toimage(images[j][0].cpu().numpy()).save(P.join(seg_dir, '%s.png' % (sn_list[idx])))
                            toimage(masks[j].cpu().numpy()).save(P.join(seg_dir, '%sagt.png' % (sn_list[idx])))
                        toimage(seg_mask.cpu().numpy()).save(P.join(seg_dir, '%s_%d_seg.png' % (sn_list[idx], i)))
                        seg_prob = F.sigmoid(seg_out[1]-seg_out[0]).cpu().numpy()
                        toimage(seg_prob, cmin=0., cmax=1.).save(P.join(seg_dir, '%s_%d_float.png' % (sn_list[idx], i)))
                        if len(cls_out.size()) == 3:
                            img = F.sigmoid(cls_out[1]-cls_out[0]).cpu().numpy()
                            img = zoom(img, 1024/img.shape[0])
                            toimage(img, cmin=0., cmax=1.).save(P.join(cls_dir, '%s_%d_mil.png' % (sn_list[idx], i)))
    for s in sm:
        s.get_metric()
    for c in cm:
        c.get_metric()
    return sm, cm





