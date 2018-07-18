#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:24:30 2018

@author: rongzhao
"""

import torch
import torch.nn as nn

def passthrough(x):
    return x

def ReLU(inplace=True):
    def nla():
        return nn.ReLU(inplace)
    return nla
    
def LeakyReLU(negative_slope=1e-2, inplace=True):
    def nla():
        return nn.LeakyReLU(negative_slope, inplace)
    return nla

def PReLU(num_parameters=1, init=0.25):
    '''num_parameter = 1 or nChannels (learn 1 parameter for each channel)'''
    def nla():
        return nn.PReLU(num_parameters, init)
    return nla

def ELU(alpha=1., inplace=True):
    '''ELU(x) = max(0,x) + min(0,α∗(exp(x)−1))'''
    def nla():
        return nn.ELU(alpha, inplace)
    return nla


def ConvDown2d(kernel=3, stride=2, padding=1, dilation=1, groups=1, bias=False):
    def down(inChans, outChans, nla=ReLU(True)):
        if nla != None:
            return nn.Sequential(
                    nn.BatchNorm2d(inChans),
                    nla(),
                    nn.Conv2d(inChans, outChans, kernel, stride, padding, dilation, groups, bias)
                    )
        else:
            return nn.Conv2d(inChans, outChans, kernel, stride, padding, dilation, groups, bias)
    return down


def MaxDown2d(kernel=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    def down(inChans, outChans, nla=ReLU(True)):
        if inChans == outChans:
            return nn.MaxPool2d(kernel, stride, padding, dilation, return_indices, ceil_mode)
        elif nla != None:
            return nn.Sequential(
                    nn.BatchNorm2d(inChans),
                    nla(),
                    nn.Conv2d(inChans, outChans, 1, 1, 0, bias=False),
                    nn.MaxPool2d(kernel, stride, padding, dilation, return_indices, ceil_mode)
                    )
        else:
            return nn.Sequential(
                    nn.Conv2d(inChans, outChans, 1, 1, 0, bias=False),
                    nn.MaxPool2d(kernel, stride, padding, dilation, return_indices, ceil_mode)
                    )
        
    return down


def DeconvUp2d(kernel=2, stride=2, padding=0, output_padding=0, groups=1, bias=False, dilation=1):
    def upsample(inChans, outChans, nla=ReLU(True)):
        if nla != None:
            return nn.Sequential(
                    nn.BatchNorm2d(inChans),
                    nla(),
                    nn.ConvTranspose2d(inChans, outChans, kernel, stride, padding, 0, groups, bias, dilation)
                    )
        else:
            return nn.ConvTranspose2d(inChans, outChans, kernel, stride, padding, 0, groups, bias, dilation)
    return upsample


def BilinearUp2d(scale_factor=2):
    def upsample(inChans, outChans, nla=ReLU(True)):
        if inChans == outChans:
            return nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        elif nla != None:
            return nn.Sequential(
                    nn.BatchNorm2d(inChans),
                    nla(),
                    nn.Conv2d(inChans, outChans, 1, 1, 0, bias=False),
                    nn.UpsamplingBilinear2d(scale_factor=scale_factor)
                    )
        else:
            return nn.Sequential(
                    nn.Conv2d(inChans, outChans, 1, 1, 0, bias=False),
                    nn.UpsamplingBilinear2d(scale_factor=scale_factor)
                    )
    return upsample


class BNNLAConv(nn.Module):
    def __init__(self, inChans, outChans, kernel, stride, padding, dilation=1, groups=1, bias=False, nla=ReLU(True)):
        super(BNNLAConv, self).__init__()
        self.model = nn.Sequential(
                nn.BatchNorm2d(inChans),
                nla(),
                nn.Conv2d(inChans, outChans, kernel, stride, padding, dilation, groups, bias)
                )
    def forward(self, x):
        return self.model(x)


class SumFusion2d(nn.Module):
    def __init__(self, inChans, skipChans, outChans, nla=ReLU(True)):
        super(SumFusion2d, self).__init__()
        self.conv1 = BNNLAConv(skipChans, outChans, 1, 1, 0, nla=nla)
        self.conv2 = BNNLAConv(inChans, outChans, 1, 1, 0, nla=nla)
        
    def forward(self, x, skipx):
        skipx = self.conv1(skipx)
        x = self.conv2(x)
        out = skipx + x
        return out


class CatFusion2d(nn.Module):
    def __init__(self, inChans, skipChans, outChans, nla=ReLU(True)):
        super(CatFusion2d, self).__init__()
        if skipChans > inChans:
            self.conv1 = BNNLAConv(skipChans, inChans, 1, 1, 0, nla=nla)
            allChans = 2 * inChans
        else:
            self.conv1 = passthrough
            allChans = skipChans + outChans
        self.conv2 = BNNLAConv(allChans, outChans, 1, 1, 0, nla=nla)
        
    def forward(self, x, skipx):
        skipx = self.conv1(skipx)
        out = torch.cat((x, skipx), 1)
        out = self.conv2(out)
        return out