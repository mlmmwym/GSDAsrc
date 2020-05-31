#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author by Elijah_Yi
from numpy import *
import numpy as np
import inputAndoutput
import cossim
from sklearn.decomposition import PCA
import csv
import random


def test(gallery, probe, gallery_label, probe_label, wfld, data_csv, group_resu, itemtime):
    """
    face verification
    :param gallery: the data of the gallery
    :param probe: the data of the probe
    :param gallery_label:  label of the gallery
    :param probe_label:  label of the probe
    :param wfld: the feature project matrix
    :param data_csv: the save path
    :param group: the save information
    :param itemtime: the itematers
    :return: the accurary
    """
    group_result = group_resu
    gallery_feature = dot(wfld.T, gallery).real
    gallery_fea_fl_name = "../%s/%d/datainfo/gallery/gallery_feature.txt" % (group_result, itemtime)
    print("gallery_feature.shape", gallery_feature.shape)
    inputAndoutput.writefile(gallery_feature, gallery_fea_fl_name)

    probe_feature = dot(wfld.T, probe).real
    probe_fea_fl_name = "../%s/%d/datainfo/probe/probe_feature.txt" % (group_result, itemtime)
    print("probe_feature.shape", probe_feature.shape)
    inputAndoutput.writefile(probe_feature, probe_fea_fl_name)

    acc = cossim.outlabel(gallery_feature, probe_feature, gallery_label, probe_label, data_csv, group_resu, itemtime)
    return acc
