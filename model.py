#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:23:56 2018

@author: rongzhao
"""

import math
import torch
import torch.nn as nn
import factory as fm

__all__ = ['uresnet_16x11x2', 
           ]

def fold_ceil(base, num_fold, ceil):
    return min(base*2**num_fold, ceil)

class ResBlock(nn.Module):
    def __init__(self, inChans, outChans, drop_rate=0.5, nla=fm.ReLU(True)):
        super(ResBlock, self).__init__()
        self.change_dim = inChans != outChans
        self.bn1 = nn.BatchNorm2d(inChans)
        self.relu1 = nla()
        self.conv1 = nn.Conv2d(inChans, outChans, 3,1,1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChans)
        self.relu2 = nla()
        self.conv2 = nn.Conv2d(outChans, outChans, 3,1,1, bias=False)
        self.do = fm.passthrough if drop_rate == 0 else nn.Dropout2d(drop_rate)
        self.projection = nn.Conv2d(inChans, outChans, 1,1,0, bias=False) \
                                   if self.change_dim else fm.passthrough

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.relu2(self.bn2(out))
        out = self.conv2(self.do(out))
        out += self.projection(x)
        return out


def make_nBlocks(nBlocks, inChans, outChans, drop_rate=0.5, nla=fm.ReLU(True), rb=ResBlock):
    # change dimension at the first convolution
    blocklist = [rb(inChans, outChans, drop_rate, nla)]
    for _ in range(nBlocks-1):
        blocklist.append(rb(outChans, outChans, drop_rate, nla))
    return nn.Sequential(*blocklist)


class UResNet_Trunk(nn.Module):
    def __init__(self, width, depth, nMod, nClass, drop_rate, downsampler, 
                 upsampler, initStride=1, nla=fm.ReLU(True), fuse='cat'):
        super(UResNet_Trunk, self).__init__()
        assert len(width)==len(depth)==len(drop_rate), 'Please check the lengths of width, depth and drop_rate.'
        self.blocks = nn.ModuleList([])
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.fusions = nn.ModuleList([])
        if fuse == 'cat':
            Fusion = fm.CatFusion2d
        else:
            Fusion = fm.SumFusion2d
        d, w = depth, width
        ew = w
        N = len(d) // 2
        self.N = N
        
        self.inconv = nn.Conv2d(nMod, w[0], 3, initStride, 1, bias=False)
        for i in range(2*N+1): # 2N+1 blocks
            self.blocks.append(make_nBlocks(d[i], w[i], w[i], drop_rate[i], nla))
            
        for i in range(N): # N down/up-samplings and fusions
            self.downs.append(downsampler(ew[i], w[i+1], nla=nla))
            self.ups.append(upsampler(ew[N+i], w[N+i+1], nla=nla))
            self.fusions.append(Fusion(w[N+i+1], ew[N-i-1], w[N+i+1], nla))
                  
    def forward(self, x):
        N = self.N
        out = self.inconv(x)
        skips, outs = [], []
        for i in range(N):
            out = self.blocks[i](out)
            skips.append(out)
            out = self.downs[i](out)
        out = self.blocks[N](out)
        for i in range(N):
            outs.append(out)
            out = self.ups[i](out)
            out = self.fusions[i](out, skips[-i-1])
            out = self.blocks[N+i+1](out)
        outs.append(out)
        return outs


class UResNet_HDS(nn.Module):
    def __init__(self, width, depth, nMod, nClass, drop_rate, mil_downfactor, downsampler, 
                 upsampler, initStride=1, nla=fm.ReLU(True), fuse='cat'):
        super(UResNet_HDS, self).__init__()
        self.trunk = UResNet_Trunk(width, depth, nMod, nClass, drop_rate, downsampler, 
                                   upsampler, initStride, nla, fuse)
        self.exit_seg = nn.ModuleList([])
        self.exit_cls = nn.ModuleList([])
        d, w = depth, width
        ew = w
        N = len(d) // 2
        self.N = N
        num_mildown = round(math.log2(mil_downfactor))
        assert num_mildown >= N, 'Invalid mil_downfactor: too small.'
        tail_len=2
                    
        for i in range(N+1): # N+1 exit pathways
            t_seg = [ResBlock(ew[N+i], ew[N+i], drop_rate[N+i], nla)]
            t_seg.append(fm.BNNLAConv(ew[N+i], nClass, 1, 1, 0, bias=True, nla=nla))
            for k in range(N-i):
                t_seg.append(upsampler(nClass, nClass, nla))
            if initStride != 1:
                t_seg.append(fm.BilinearUp2d(initStride)(nClass, nClass, nla))
            self.exit_seg.append(nn.Sequential(*t_seg))
            
            t_cls = []
            down_order = num_mildown-N+i
            if down_order <= tail_len:
                for j in range(down_order):
                    inChans, outChans = fold_ceil(ew[N+i],j,ew[N]), fold_ceil(ew[N+i],j+1,ew[N])
                    t_cls += [downsampler(inChans, outChans, nla),
                              ResBlock(outChans, outChans, drop_rate[N+i], nla)]
                for j in range(down_order, tail_len):
                    inChans, outChans = fold_ceil(ew[N+i],j,ew[N]), fold_ceil(ew[N+i],j+1,ew[N])
                    t_cls.append(ResBlock(inChans, outChans, drop_rate[N+i], nla))
                t_cls.append(fm.BNNLAConv(fold_ceil(ew[N+i],tail_len,ew[N]), nClass, 1, 1, 0, bias=True, nla=nla))
            else:
                for j in range(tail_len):
                    inChans, outChans = fold_ceil(ew[N+i],j,ew[N]), fold_ceil(ew[N+i],j+1,ew[N])
                    t_cls += [downsampler(inChans, outChans, nla),
                              ResBlock(outChans, outChans, drop_rate[N+i], nla)]
                t_cls += [nn.AvgPool2d(2**(down_order-tail_len)),
                          fm.BNNLAConv(fold_ceil(ew[N+i],tail_len,ew[N]), nClass, 1, 1, 0, bias=True, nla=nla)]
            self.exit_cls.append(nn.Sequential(*t_cls))
        
    def forward(self, x):
        N = self.N
        outs = self.trunk(x)
        segs, clses = [], []
        for i in range(N+1):
            seg = self.exit_seg[i](outs[i])
            segs.append(seg)
            cls = self.exit_cls[i](outs[i])
            clses.append(cls)
        
        return torch.stack(segs), torch.stack(clses)

def uresnet_16x11x2(nMod, nClass, drop_rate, mil_downfactor, nla=fm.ReLU(True),
                          downsampler=fm.MaxDown2d(), upsampler=fm.DeconvUp2d()):
    model = UResNet_HDS((16,32,64,128,128,128,128,128,64,32,16), (2,)*11, nMod, nClass, 
                       drop_rate, mil_downfactor, downsampler, upsampler, nla=nla)
    return model


    
    
