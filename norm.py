#! /usr/bin/env python3
import math as _m
import numpy as np

def l1(x):
    """
    compute L1 norm of input
    x:
        numpy.ndarray
    return:
        the L1 norm of input
    """
    if x.ndim == 1:
        return np.abs(x).sum()
    elif x.ndim == 2:
        return np.max(np.sum(np.abs(x), axis=0))
def l2(x):
    """
    compute L2 norm of input
    x:
        numpy.ndarray
    return:
        the L2 norm of input
    """
    return _m.sqrt((x*x).sum())

def row(x):
    """
    compute the row norm of input
    x:
        numpy.ndarray
    return:
        the row norm of input
    """
    if x.ndim == 1:
        return np.max(np.abs(x))
    elif x.ndim == 2:
        return np.max(np.sum(np.abs(x), axis=1))

