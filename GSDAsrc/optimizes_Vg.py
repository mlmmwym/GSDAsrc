#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author by Elijah_Yi

from numpy import  *
from sklearn import linear_model
import math
import numpy as np



def loadMat(filename):
    mat = np.loadtxt(filename)
    return  mat

def optimizeVg(p,q_mean):
    lambd_ = 0.05
    tau =5
    mat_coef = np.array([])
    coef_mat = np.zeros((q_mean.shape[1], p.shape[1]), dtype=np.float32)
    ht = math.sqrt(lambd_)*np.ones((q_mean.shape[1],1), np.float32)
    q_new = np.r_[q_mean, ht.T]
    print('q_mean.shape',q_mean.shape)
    print('q_new.shape',q_new.shape)
    max = np.zeros((p.shape[1], 1), dtype=np.float32)
    for i in range(p.shape[1]):
        pi = p[:,i]
        pi_new = np.r_[pi, math.sqrt(lambd_)]
        clf = linear_model.Lars(n_nonzero_coefs=tau)#n_nonzero_coefs  限制非零元素个数为1
        clf.fit(q_new.real, pi_new.real)#利用LARS模型解决Vs  Vt 参数
        mat_coef=clf.coef_
        coef_mat[:, i] = mat_coef
        coef_abs= np.abs(coef_mat[:,i])#取绝对值
        coef_abs_n = coef_abs
        coef_abs_n.shape = ((q_mean.shape[1],1))
        max[i] = coef_abs[np.argmax(coef_abs)]
        ht = ht - ((1. / (2 * max[i])) * coef_abs)
    return coef_mat

def writefile(mat,filename):
    mat = np.array(mat)
    with open (filename,'w') as fi:
        for i in range(mat.shape[0]):
            for j in range((mat.shape[1])):
                fi.write(str(mat[i][j])+'    ')
            fi.write('\n')
            i+=1
        file.close(fi)
    print "Write success！"
