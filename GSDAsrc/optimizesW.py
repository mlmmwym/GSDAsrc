#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author by Elijah_Yi

from __future__ import division
from numpy import *
import numpy as np
import math


def optimizesW(Xs, Xt, Vs, Vt, W_dim=300):
    """
    optimization the project matrix W
    :param Xs: source domian data
    :param Xt: target domian data
    :param Vs: the coefficient of source domian
    :param Vt: the coefficient of target domian
    :param W_dim: prior experience dimension of feature
    :return: project matrix W
    """
    # print Xs.shape
    # print Xt.shape
    ns = Xs.shape[1]
    nt = Xt.shape[1]
    Xs = Xs - np.mean(Xs)
    Xt = Xt - np.mean(Xt)
    Xs_d = dot(Xs, Xs.T)
    Xt_d = dot(Xt, Xt.T)
    Xs_zero = np.zeros((Xs.shape[0], Xt.shape[0]), dtype=np.float32)
    Xt_zero = np.zeros((Xt.shape[0], Xs.shape[0]), dtype=np.float32)
    Xs_a = (1. / ns) * Xs_d
    Xt_a = (1. / nt) * Xt_d
    sigma_b = np.bmat("Xs_a Xs_zero;Xt_zero Xt_a")
    Vs_d = dot(Vs, Vs.T)
    Vt_d = dot(Vt, Vt.T)
    Ins = np.eye(ns, dtype=np.float32)
    Int = np.eye(nt, dtype=np.float32)
    # print Ins.shape,Xs.shape,Vt.shape
    sigma_w_A = dot(dot(Xs, (Ins / ns) + (Vt_d / nt)), Xs.T)
    sigma_w_B = (-1) * dot(dot(Xs, ((Vs.T / ns) + (Vt / nt))), Xt.T)
    sigma_w_C = (-1) * dot(dot(Xt, ((Vt.T / nt) + (Vs / ns))), Xs.T)
    sigma_w_D = dot(dot(Xt, (Int / nt) + (Vs_d / ns)), Xt.T)
    sigma_w = np.bmat("sigma_w_A sigma_w_B;sigma_w_C sigma_w_D")
    sigma_w_inv = np.linalg.pinv(sigma_w)

    W_vect = dot(sigma_w_inv, sigma_b)
    eigVals, eigVects = np.linalg.eig(W_vect)
    if len(eigVals) <= W_dim:
        W_dim = len(eigVals)
    eigVects = np.array(eigVects)
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(W_dim + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    W_a = n_eigVect
    W_a = np.array(W_a).real
    Ws = np.zeros([Xs.shape[0], W_dim], dtype="float32")
    Wt = np.zeros([Xt.shape[0], W_dim], dtype="float32")
    Ws[:, :] = W_a[0:Xs.shape[0]]
    Wt[:, :] = W_a[Xs.shape[0]:]
    # calculate trace
    Trace_b = 0.
    T_b = dot(dot(W_a.T, sigma_b), W_a)
    for i in range(T_b.shape[0]):
        Trace_b += T_b[i, i]
    # Trace of intra-class scatter matrix
    T_w = dot(dot(W_a.T, sigma_w), W_a)
    Trace_w = 0.
    for i in range(T_w.shape[0]):
        Trace_w += T_w[i, i]
    # Traces of inter-class scatter matrix
    Trace_a = 0.
    T_a = dot(T_b, np.linalg.pinv(T_w))
    for i in range(T_a.shape[0]):
        Trace_a += T_a[i, i]
    # Ratio of traces
    T = Trace_b / Trace_w
    T = T.real
    return Ws, Wt, T, Trace_a.real


def normalizeW(matrix):
    """
    generate the l1 normal project matrix of W
    :param matrix:
    :return:
    """
    suMat = np.zeros((1, matrix.shape[1]))
    # 将每一列元素的平方和累加
    for i in range(matrix.shape[1]):
        suMat[0, i] = np.linalg.norm(matrix)  # dot(matrix[:,i],matrix[:,i].T)
        # for j in range(inMat.shape[0]):
        #     suMat[0, i] += inMat[j][i]*inMat[j][i]
    W = np.zeros(matrix.shape, dtype='float64')
    # 每列的元素除以自身列的平方和的根号
    for i in range(matrix.shape[1]):
        W[:, i] = matrix[:, i] / (np.sqrt(suMat[:, i]))
        # for j in range(matrix.shape[0]):
        #     W[j,i] = float(matrix[j,i]/(np.sqrt(suMat[:,i])))
    return W
