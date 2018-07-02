import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR)) # config
import config
from caffe import layers as L, params as P


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),
        param=[dict(lr_mult=1), dict(lr_mult=2)])
    return conv, L.ReLU(conv, in_place=True)

def conv(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, weight_filler=dict(type='msra'), bias_filler=dict(type='constant'),
        param=[dict(lr_mult=1), dict(lr_mult=2)])
    return conv

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fc(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout,
        param=[dict(lr_mult=1), dict(lr_mult=2)],
        weight_filler=dict(type='gaussian', std=0.001),
        bias_filler=dict(type='constant'))
    return fc

def fc_relu_dropout(bottom, nout, dropout):
    fc = L.InnerProduct(bottom, num_output=nout,
        param=[dict(lr_mult=1), dict(lr_mult=2)],
        weight_filler=dict(type='gaussian', std=0.001),
        bias_filler=dict(type='constant'))
    return fc, L.ReLU(fc, in_place=True), L.Dropout(fc, dropout_ratio=dropout, in_place=True)