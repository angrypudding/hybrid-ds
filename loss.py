#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:31:25 2018

@author: rongzhao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def map2label(input):
    input = input.view(input.size(0), input.size(1), -1) # Size: NxCxHxW -> NxCxP
    # eigen patch position: the position with smallest background probability
    p = torch.min(input[:,0,:], dim=1)[1]
    p = p.detach()
    predicted = torch.stack([input[i,:,p[i]] for i in range(len(p))]) # NxC
    return predicted

class MultiOutputLoss(object):
    def __init__(self, loss_func, loss_weight, device, decay_factor=1):
        self.loss_func = loss_func
        self.loss_weight = torch.FloatTensor(loss_weight).to(device)
        self.decay_factor = decay_factor
        
    def __call__(self, input, target):
        loss = 0
        loss_arr = []
        for i in range(len(input)):
            loss_so = self.loss_func(input[i], target)
            loss_arr.append(loss_so)
            loss += self.loss_weight[i] * loss_so
            
        return loss, loss_arr
        
    def change_loss_weight(self, loss_weight):
        self.loss_weight = loss_weight
                
    def decay_loss_weight(self, decay_factor=None):
        df = decay_factor if decay_factor else self.decay_factor
        for i in range(len(self.loss_weight)-1):
            self.loss_weight[i] *= df
            
    def decay_loss_weight_epoch(self, epoch, decay_factor=None):
        df = decay_factor if decay_factor else self.decay_factor
        for i in range(len(self.loss_weight)-1):
            self.loss_weight[i] *= df
            

class SparseMILLoss(object):
    def __init__(self, sparse_factor, weight=None, size_average=True, ignore_index=-100, reduce=True):
        self.sparse_factor = sparse_factor
        self.CELoss = nn.CrossEntropyLoss(torch.tensor(weight), size_average, ignore_index, reduce)
        
    def __call__(self, input, target):
        input = input.view(input.size(0), input.size(1), -1) # Size: NxCxHxW -> NxCxP
        prob_input = F.softmax(input, dim=1) # Size: NxCxP
        sparse_penalty = torch.sum(prob_input[:,1:,:]) / input.size(0)
        predicted = map2label(input) # NxC
        ce_loss = self.CELoss(predicted, target)
        loss = ce_loss + self.sparse_factor * sparse_penalty
        return loss
            
            
            
            
            
