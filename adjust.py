#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:58:27 2017

@author: zhang
"""
import math

def adjust_lr(i, epoch, max_epoch, **kw):
    lr_policy = kw.get('lr_policy')
    base_lr = kw.get('base_lr')
    lr_policy = lr_policy[i] if isinstance(lr_policy, (tuple, list)) else lr_policy
    base_lr = base_lr[i] if isinstance(base_lr, (tuple, list)) else base_lr
    if lr_policy == 'fixed':
        return base_lr
    elif lr_policy == 'step':
        gamma = kw.get('gamma')
        stepsize = kw.get('stepsize')
        return base_lr * gamma ** math.floor(epoch/stepsize)
    elif lr_policy == 'exp':
        gamma = kw.get('gamma')
        return base_lr * gamma ** epoch
    elif lr_policy == 'inv':
        gamma = kw.get('gamma')
        power = kw.get('power')
        return base_lr * (1 + gamma * epoch) ** (-power)
    elif lr_policy == 'multistep':
        gamma = kw.get('gamma')
        stepvalue = kw.get('stepvalue')
        lr = base_lr
        for value in stepvalue:
            if epoch >= value:
                lr *= gamma
            else:
                break
        return lr
    elif lr_policy == 'poly':
        power = kw.get('power')
        return base_lr * (1 - epoch / max_epoch) ** power
    elif lr_policy == 'sigmoid':
        gamma = kw.get('gamma')
        stepsize = kw.get('stepsize')
        return base_lr * (1 / (1 + math.exp(-gamma * (epoch - stepsize))))
    else:
        raise RuntimeError('Unknown lr_policy: %s' % lr_policy)
        
def adjust_opt(optimizer, epoch, max_epoch, **kw):
    for i, param in enumerate(optimizer.param_groups):
        param['lr'] = adjust_lr(i, epoch, max_epoch, **kw)




        
        