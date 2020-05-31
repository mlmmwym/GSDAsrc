#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author by Elijah_Yi
from numpy import *
import numpy as np
import os


# Calculate  cosine similarity
def calCos(vector_1, vector_2):
    """
    Calculate  cosine similarity
    :param vector_1: vector
    :param vector_2: vector
    :return: similar
    """
    num = float(dot(vector_1.T, vector_2))  # 若是为行向量则为A*B.T
    mold = (linalg.norm(vector_1)) * (linalg.norm(vector_2))
    cos = num / mold
    sim = 0.5 + 0.5 * cos  # 归一化
    # print cos
    return sim


def outlabel(gallery, probe, galleryLabel, probeLabel, csv_path, group_resu, itemtime):
    """
    predict the probe  label
    :param gallery: the feature of the gallery
    :param probe: the feature of the probe
    :param galleryLabel: the label of the gallery
    :param probeLabel: the label of the probe
    :param csv_path: csv path
    :param group: the time of the experiments
    :param itemtime: the itemater of the experiments
    :return: accurary
    """
    count1 = 0.
    count2 = 0.
    group_result = group_resu
    acc_fl_name = '../%s/%d/datainfo/parameter/acc.csv' % (group_resu, itemtime)
    probe_csv = []
    for row in (open(csv_path)):
        row, iv = row.split('\n')
        probe_csv.append(row)
    for i in range(probe.shape[1]):
        cosmax = []
        for j in range(gallery.shape[1]):
            sim = calCos(probe[:, i], gallery[:, j])
            cosmax.append(sim)
        cosmax = np.array(cosmax)
        ind = np.argmax(cosmax)
        if galleryLabel[ind] == probeLabel[i]:
            count1 += 1
        else:
            count2 += 1
    acc = count1 / (count1 + count2)
    print("acc", acc)
    try:
        wri = open(acc_fl_name, 'w')
        wri.write(str(acc))
        wri.close()
    except IOError:
        acc_fl_path = os.path.dirname(acc_fl_name)
        os.makedirs(acc_fl_path)
        wri = open(acc_fl_name, 'w')
        wri.write(str(acc))
        wri.close()
    return acc
