#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
'''
from numpy import *
import numpy as np
import inputAndoutput
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def between_class_S_B(data, label_):
    """
    calculate the inter-class scatter matrix
    :param data:raw data
    :param label: data of label
    :return: inter-class scatter matrix
    """
    label = label_
    label.shape = (1, len(label))
    classes = np.unique(label)
    clusters = len(classes)
    count = 0
    within_class = np.zeros(data.shape[0], dtype="float32")
    i = 0
    num_label = []
    with_in_class_mean = []  # 类内均值矩阵
    while i < data.shape[1]:
        j = i
        n_sum = np.zeros(data.shape[0], dtype="float32")
        # for j in range(i,y.shape[1]):
        while j < data.shape[1]:
            if label[0][j] == label[0][i]:
                n_sum += data[:, j]
                count += 1
                if j == data.shape[1] - 1:
                    num_label.append(count)
                    within_class = n_sum / count
                    with_in_class_mean.append(within_class)
                    i = j + 1
                    break
                j += 1
            else:
                num_label.append(count)
                within_class = n_sum / count
                with_in_class_mean.append(within_class)
                i = j
                count = 0
                break
    with_in_class_mean = np.array(with_in_class_mean).T
    all_mean = np.mean(data, axis=1)
    all_mean.shape = (len(all_mean), 1)
    S_B = np.zeros((data.shape[0], data.shape[0]))
    temp = np.zeros_like(all_mean)
    for i in range(clusters):
        temp = with_in_class_mean[:, i]
        temp.shape = (len(temp), 1)
        S_B += num_label[i] * (temp - all_mean).dot((temp - all_mean).T)
    return S_B


def statistics(X, y):
    """
    calculate the intra-class means
    :param X:ndarray:shape(featrue,sample)
    :param y:(label)
    :return:class number,nij. means of intra-class
    """
    y.shape = (len(y))
    nij = np.bincount(y)
    within_class = []
    start = 0
    end = 0
    for i in range(nij.shape[0]):
        end += nij[i]
        within_class.append(np.mean(X[:, start:end], axis=1))
        start = end

    within_class_mean = np.vstack(within_class).T

    return within_class_mean, nij


def calc_Sw(data, label, with_class_mean, each_class_sample_num, vg):
    '''
    calculate intra-class scatter matrix
    :param data: input data
    :param label: input label
    :param with_class_mean: means of input class
    :param class_num: num of input class
    :return:
    '''
    n_ecs = each_class_sample_num
    classes = np.unique(label)
    clusters = len(classes)
    temp = np.zeros((with_class_mean.shape[0], 1))
    mean_temp = np.zeros_like(with_class_mean)
    S_W = np.zeros((data.shape[0], data.shape[0]))
    ind = 0
    l = np.array([])
    for i in range(vg.shape[1]):
        l = vg[:, i]
        for j in range(vg.shape[0]):
            if l[j] != 0:
                index_ = j
                for k in range(n_ecs[j]):
                    temp = data[:, sum(n_ecs[:j]) + k]
                    temp.shape = (len(temp), 1)
                    mean_temp = with_class_mean[:, j]
                    mean_temp.shape = (len(with_class_mean[:, j]), 1)
                    # print 'temp\n',temp
                    # print 'mean_temp\n',mean_temp
                    # print 'with_class_mean[:,j].shape\n',with_class_mean[:,j]
                    # print '(temp - with_class_mean[:,j])\n',(temp - mean_temp)
                    # print '(temp - mean_temp).dot((temp - mean_temp).T)\n',(temp - mean_temp).dot((temp - mean_temp).T)
                    S_W += (temp - mean_temp).dot((temp - mean_temp).T) * vg[j, i]
                    # print 'S_W\n',S_W
    return S_W


def within_class_S_T(data):
    '''
    calculate otal divergence matrix
    :param data: input data to calculate the total divergence matrix
    :return: total Total divergence matrix
    '''
    S_W = np.zeros((data.shape[0], data.shape[0]))
    all_mean_w = np.mean(data, axis=1)
    all_mean_w.shape = (len(all_mean_w), 1)
    temp = np.zeros_like(all_mean_w)
    for i in range(data.shape[1]):
        temp = data[:, i]
        temp.shape = (len(temp), 1)
        S_W += (temp - all_mean_w).dot((temp - all_mean_w).T)
    return S_W


def calc_gallery_S_T(g_data, total_data):
    S_T = np.zeros((g_data.shape[0], g_data.shape[0]))
    all_mean_w = np.mean(total_data, axis=1)
    all_mean_w.shape = (len(all_mean_w), 1)
    temp = np.zeros_like(all_mean_w)
    for i in range(g_data.shape[1]):
        temp = g_data[:, i]
        temp.shape = (len(temp), 1)
        S_T += (temp - all_mean_w).dot((temp - all_mean_w).T)
    return S_T


def lda(data, tar_gal, sb_g, label, g_label):
    """
    calculate the final project matrix
    :param data: targetized source domian
    :param tar_gal: gallery and target domian data
    :param sb_g: estimated inter class scatter matrix of gallery
    :param label: source label
    :param g_label: gallery label
    :return: project matrix
    """
    classes_s = np.unique(label)
    classes_g = np.unique(g_label)
    fea_num = len(classes_s) + len(classes_g)
    S_T = within_class_S_T(tar_gal)
    S_B_1 = between_class_S_B(data, label)
    S_B = S_B_1 + sb_g
    S_W_inv = np.linalg.inv(S_T)
    W_vect = dot(S_W_inv, S_B)
    eigVals, eigVects = np.linalg.eig(W_vect)
    eigVects = np.array(eigVects)
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(fea_num):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    W = n_eigVect
    return W.real
