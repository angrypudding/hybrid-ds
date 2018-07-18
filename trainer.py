#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:41:48 2018

@author: rongzhao
"""

import os
import os.path as P
import torch
import numpy as np

from adjust import adjust_opt
from misc import timestr
from validate import validate_cs

class Tester(object):
    '''A functional class facilitating and supporting all procedures in training phase, based on Trainer class'''
    def __init__(self, model_cube, data_cube, snapshot_scheme, device):
        self._trainer = Trainer(model_cube, data_cube, None, None, None, 
                                snapshot_scheme, device, True)
    def test(self, state_suffix, foldername, is_indiv=False, is_save_png=False):
        self._trainer.test(state_suffix, foldername, is_indiv, is_save_png)

class Trainer(object):
    '''A functional class facilitating and supporting all procedures in training phase'''
    def __init__(self, model_cube, data_cube, criterion_cube, lr_scheme, 
                 snapshot_scheme, device, wrap_test=False, clsfromseg=False):
        self.model, self.optimizer, self.start_epoch, self.num_mo = \
            self.init_model(model_cube) # Initialize model
#        if model_cube['resume']:
#            self._optim_device(device)
        self.parse_dataloader(data_cube)
        self.parse_criterion(criterion_cube)
        self.lr_scheme = lr_scheme
        self.snapshot_scheme = snapshot_scheme
        self.max_epoch = lr_scheme['max_epoch'] if not wrap_test else 0
        self.root = snapshot_scheme['root']
        self.device = device
        self.clsfromseg = clsfromseg
        
        if not wrap_test:
            with open( P.join(self.root, 'description.txt'), 'w' ) as f:
                f.write(str(lr_scheme) + '\n' + str(snapshot_scheme) + '\n' + str(self.model))
        self.model.to(self.device)
        
    def train(self):
        '''Cordinate the whole training phase, mainly recording of losses and metrics, 
        lr and loss weight decay, snapshotting, etc.'''
        loss_all = []
        max_seg_metric, max_cls_metric = 0, 0
        lossF = open(P.join(self.root, 'loss.txt'), 'a')
        print(timestr(), 'Optimization Begin')
        for epoch in range(self.start_epoch, self.max_epoch+1):
            # Adjust learning rate
            adjust_opt(self.optimizer, epoch-1, **self.lr_scheme)
            loss_dict = self.train_epoch()
            loss_all.append(loss_dict['loss'])
            lossF.write('%d,%.7f\n' % (epoch, loss_all[-1]))
            lossF.flush()
            
            if epoch % self.snapshot_scheme['display_interval'] == 0 or epoch == self.start_epoch:
                N = self.snapshot_scheme['display_interval']
                loss_avg = np.array(loss_all[-N:]).mean()
                first_epoch = epoch if epoch == self.start_epoch else epoch+1-N
                print('%s Epoch %d ~ %d: loss = %.7f, current lr = %.7e' %
                      (timestr(), first_epoch, epoch, loss_avg, self._get_lr()))
            
            if epoch % self.snapshot_scheme['snapshot_interval'] == 0 or epoch == self.start_epoch:
                self._snapshot(epoch)
            
            if epoch % self.snapshot_scheme['test_interval'] == 0 or epoch == self.start_epoch:
                metric_dict = self.validate_online()
                if max_seg_metric <= metric_dict['val/seg_dsc'] and epoch > 10:
                    max_seg_metric = metric_dict['val/seg_dsc']
                    self._snapshot(epoch, 'seg_max')
                if max_cls_metric <= metric_dict['val/cls_acc'] and epoch > 10:
                    max_cls_metric = metric_dict['val/cls_acc']
                    self._snapshot(epoch, 'cls_max')
                    
            self.criterion_seg.decay_loss_weight()
            self.criterion_cls.decay_loss_weight()
        
        self._snapshot(epoch, str(epoch))
        lossF.close()
        
    def train_epoch(self):
        '''Train the model for one epoch, loss information is recorded'''
        self.model.train()
        loss_buf, loss_seg_buf, loss_cls_buf, loss_seg_arr_buf, loss_cls_arr_buf = [],[],[],[],[]
        for images, masks, labels in iter(self.trainloader):
            images, masks, labels = images.to(self.device), masks.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            seg_output, cls_output = self.model(images)
            loss_seg, loss_seg_arr = self.criterion_seg(seg_output, masks)
            loss_cls, loss_cls_arr = self.criterion_cls(cls_output, labels)
            loss = self.segcls_weight[0]*loss_seg + self.segcls_weight[1]*loss_cls
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.item())
            loss_seg_buf.append(loss_seg.item())
            loss_cls_buf.append(loss_cls.item())
            loss_seg_arr_buf.append([v.item() for v in loss_seg_arr])
            loss_cls_arr_buf.append([v.item() for v in loss_cls_arr])
        loss_dict = self.format_loss_buffer(loss_buf, loss_seg_buf, loss_cls_buf, loss_seg_arr_buf, loss_cls_arr_buf)
        
        return loss_dict
        
    def test(self, state_suffix, foldername, is_indiv=False, is_save_png=False, 
             seg_thres=0.5, cls_thres=0.5):
        '''Cordinate the testing of the model after training'''
        save_dir = P.join(self.root, foldername)
        pretrain = P.join(self.root, 'state_%s.pkl'%state_suffix)
        self._load_pretrain(pretrain)
        self.validate_final(save_dir, is_indiv, is_save_png, seg_thres, cls_thres)
        
    def validate_final(self, save_dir, is_indiv, is_save_png, seg_thres=0.5, 
                       cls_thres=0.5):
        '''Validate the model after training finished, detailed metrics would be recorded'''
        def validate_split(dataloader, sn_list, split):
            png_dir = P.join(save_dir, split) if is_save_png else None
            sm_arr, cm_arr = \
            validate_cs(self.model, dataloader, sn_list, self.device, self.num_mo, 
                        png_dir, seg_thres, cls_thres, True, clsfromseg=self.clsfromseg)
            max_name = save_dir.split('/')[-1]
            sm_arr[-1].print_metric(max_name+ '-' + split)
            cm_arr[-1].print_metric(max_name+ '-' +split)
            
            split_segF = open(P.join(save_dir, '%s_seg.txt' % split), 'w')
            split_clsF = open(P.join(save_dir, '%s_cls.txt' % split), 'w')
            for i in range(-1, -self.num_mo-1, -1):
                sm, cm = sm_arr[i], cm_arr[i]
                sm.write_metric(split_segF, 'Output %d:'%i, is_indiv)
                cm.write_metric(split_clsF, 'Output %d:'%i, is_indiv)
            split_segF.close()
            split_clsF.close()
        
        os.makedirs(save_dir, exist_ok=True)
        validate_split(self.trainseqloader, self.train_sn, 'train')
        validate_split(self.valloader, self.val_sn, 'val')
        validate_split(self.testloader, self.test_sn, 'test')
                
    def validate_online(self):
        '''Validate the model during training, record a minimal number of metrics'''
        def validate_split(dataloader, sn_list, split):
            sm_arr, cm_arr = \
            validate_cs(self.model, dataloader, sn_list, self.device, self.num_mo, None,
                        clsfromseg=self.clsfromseg)
            
            return sm_arr, cm_arr
        
        metric_dict = dict()
        def convert_metric_dict(prefix, sm_arr, cm_arr):
            '''Convert metrics in the last output'''
            for k, v in sm_arr[-1].metric.items():
                metric_dict['%s/seg_%s' % (prefix, k)] = v
            for k, v in cm_arr[-1].metric.items():
                metric_dict['%s/cls_%s' % (prefix, k)] = v
                      
        sm_arr_train, cm_arr_train = \
        validate_split(self.trainseqloader, self.train_sn, 'Train')
        convert_metric_dict('train', sm_arr_train, cm_arr_train)
        
        sm_arr_val, cm_arr_val = \
        validate_split(self.valloader, self.val_sn, 'Validation')
        convert_metric_dict('val', sm_arr_val, cm_arr_val)
        
        return metric_dict
        
    @staticmethod
    def format_loss_buffer(loss_buf, loss_seg_buf, loss_cls_buf, loss_seg_arr_buf, loss_cls_arr_buf):
        '''Gather all different losses into a dictionary with clear-named keys'''
        loss = np.array(loss_buf).mean()
        loss_seg = np.array(loss_seg_buf).mean()
        loss_cls = np.array(loss_cls_buf).mean()
        loss_seg_arr = np.array(loss_seg_arr_buf).mean(0)
        loss_cls_arr = np.array(loss_cls_arr_buf).mean(0)
        loss_dict = dict()
        loss_dict['loss'] = loss
        loss_dict['loss_seg'] = loss_seg
        loss_dict['loss_cls'] = loss_cls
        for i in range(-1, -len(loss_seg_arr)-1, -1):
            loss_dict['loss_seg/%d' % i] = loss_seg_arr[i]
        for i in range(-1, -len(loss_cls_arr)-1, -1):
            loss_dict['loss_cls/%d' % i] = loss_cls_arr[i]
        return loss_dict
        
        
    @staticmethod
    def init_model(model_cube):
        '''Initialize the model, optimizer and related variables according to model_cube'''
        model = model_cube['model']
        optimizer = model_cube['optimizer']
        pretrain = model_cube['pretrain']
        resume = model_cube['resume']
        num_mo = model_cube['num_mo']
        start_epoch = 1
        if resume:
            if os.path.isfile(resume):
                model.cpu()
                state = torch.load(resume)
                model.load_state_dict(state['state_dict'])
                optimizer.load_state_dict(state['optimizer'])
                start_epoch = state['epoch'] + 1
            else:
                raise RuntimeError('No checkpoint found at %s' % pretrain)
        elif pretrain:
            if os.path.isfile(pretrain):
                model.cpu()
                state = torch.load(pretrain)
                model.load_state_dict(state['state_dict'])
            else:
                raise RuntimeError('No checkpoint found at %s' % pretrain)
        else:
            weight_init_func = model_cube['init_func']
            model.apply(weight_init_func)
        return model, optimizer, start_epoch, num_mo
    
    def parse_dataloader(self, data_cube):
        self.trainloader = data_cube.trainloader()
        self.valloader= data_cube.valloader()
        self.testloader = data_cube.testloader()
        self.trainseqloader = data_cube.trainseqloader()
        self.val_sn = data_cube.val_sn()
        self.test_sn = data_cube.test_sn()
        self.train_sn = data_cube.train_sn()
        
    def parse_criterion(self, criterion_cube):
        if criterion_cube is None:
            return
        self.criterion_seg = criterion_cube['criterion_seg']
        self.criterion_cls = criterion_cube['criterion_cls']
        self.segcls_weight = criterion_cube['segcls_weight']
        
    def _load_pretrain(self, pretrain):
#        self.model.cpu()
        state = torch.load(pretrain)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
    
    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
        
    def _snapshot(self, epoch, name=None):
        '''Take snapshot of the model, save to root dir'''
#        self.model.to(torch.device('cpu'))
        state_dict = {'epoch': epoch,
                      'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        if name is None:
            filename = '%s/state_%04d.pkl' % (self.root, epoch)
        else:
            filename = '%s/state_%s.pkl' % (self.root, name)
        print('%s Snapshotting to %s' % (timestr(), filename))
        torch.save(state_dict, filename)
#        self.model.to(self.device)
        
    def _optim_device(self, device):
        for k, v in self.optimizer.state.items():
            for kk, vv in v.items():
                v[kk] = vv.to(device)
                
        
        
        
        
        
        
        