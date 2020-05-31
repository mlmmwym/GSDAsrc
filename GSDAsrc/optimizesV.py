#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author by Elijah_Yi

from numpy import *
from sklearn import linear_model
import math
import numpy as np


def loadMat(filename):
    mat = np.loadtxt(filename)
    return mat


def optimizeV(sourceMat, targetMat, sourceWeights, targetWeights):
    """
    optimization the coefficient V
    :param sourceMat: source domian data
    :param targetMat: target domian data
    :param sourceWeights: weight of source domian
    :param targetWeights: weight of target domian
    :return: V
    """
    lambd_ = 0.05
    tau = 3
    Zt = dot(targetWeights.T, targetMat)
    ht = math.sqrt(lambd_) * np.ones((targetMat.shape[1], 1), np.float32)
    coef_abs = np.zeros((targetMat.shape[1], 1), dtype=np.float32)
    coef_abs_n = np.zeros((targetMat.shape[1], 1), np.float32)
    Zt_n = np.r_[Zt, ht.T]
    Zt_n = np.array(Zt_n)
    Zs = dot(sourceWeights.T, sourceMat)
    # coef_mat
    coef_mat = np.zeros((targetMat.shape[1], sourceMat.shape[1]), dtype=np.float32)
    max = np.zeros((sourceMat.shape[1], 1), dtype=np.float32)
    for i in range(sourceMat.shape[1]):
        zi_s = Zs[:, i]
        zi_s_n = np.r_[zi_s, math.sqrt(lambd_)]
        zi_s_n.shape = (len(zi_s_n), 1)
        clf = linear_model.Lars(n_nonzero_coefs=tau)  # tau :
        clf.fit(Zt_n.real, zi_s_n.real)  # used LARS to solve the parameter of Vs,Vt
        mat_coef = clf.coef_
        coef_mat[:, i] = mat_coef
        coef_abs = np.abs(coef_mat[:, i])
        coef_abs_n = coef_abs
        coef_abs_n.shape = ((targetMat.shape[1], 1))
        max[i] = coef_abs[np.argmax(coef_abs)]
        ht = ht - ((1. / (2 * max[i])) * coef_abs)
    return coef_mat