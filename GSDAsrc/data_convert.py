#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author by Elijah_Yi
import os
import creat_csv
import numpy as np
import pictomat
import inputAndoutput
from sklearn.decomposition import PCA
import random


def skpca(data, data_g, data_p, k=0.98):
    """
    Dimensionality reduction by PCA
    :param data: raw data
    :param data_g: data of gallery
    :param data_p: data of probe
    :param k: Variance contribution rate
    :return: transfer data
    """
    newdata = data.T
    newdata = np.array(newdata)
    g_new_data = data_g.T
    p_new_data = data_p.T
    pca = PCA(n_components=k)
    pca.fit(newdata)
    tran_g = pca.transform(g_new_data)
    tran_p = pca.transform(p_new_data)
    return tran_g.T, tran_p.T


def s_pca(data, k=0.98):
    """
    Dimension reduction of single data
    :param data: raw data
    :param k: Variance contribution rate
    :return: transfer data
    """
    newdata = data.T
    pca = PCA(n_components=k)
    redata = pca.fit_transform(newdata)
    return redata.T


def create_pca_data(sourceDomainPath_0, sourceDomainCsv_0, feature_num, data_raw_name,
                    data_raw_label_name, data_csv, domain_name, domain_label_name, num_feature, randomSelect=7):
    '''
    :param sourceDomainPath_0: source domain image path
    :param sourceDomainCsv_0: source domain CSV path
    :param feature_num: raw sample
    :param data_raw_name: raw sample data matrix
    :param data_raw_label_name: raw sample label matrix
    :param data_csv: CSV file of selected sample
    :param num_feature: dimension reduction scale / dimension reduced
    :param domain_name:  selected domain samples
    :param domain_label_name: selected sample label
    :param num_feature: retain sample dimension percentage
    :return: dimension reduction matrix
    '''
    creat_csv.creat_csv(sourceDomainPath_0, sourceDomainCsv_0)
    data_raw, data_raw_label = pictomat.imagetovec(sourceDomainCsv_0, feature_num)
    inputAndoutput.writefile(data_raw, data_raw_name)
    inputAndoutput.writefile(data_raw_label, data_raw_label_name)
    data_csv_0 = []
    for row in (open(sourceDomainCsv_0)):
        row, iv = row.split('\n')
        data_csv_0.append(row)
    domain, domain_label, data_csv_0 = rand_sample(data_raw, data_raw_label, data_csv_0, randomSelect)
    with open(data_csv, 'w') as fi:
        for row in data_csv_0:
            fi.write(str(row) + '\n')
        fi.close()
    inputAndoutput.writefile(domain, domain_name)
    inputAndoutput.writefile(domain_label, domain_label_name)
    domain_pca = s_pca(domain, k=num_feature)
    return domain, domain_pca, domain_label


def creat_domain(group_data_info, itemtime):
    '''
    generate training data
    :param group_data_info: data save path
    :param group_resu: the save path of training result
    :param itemtime: iteration of training
    :return: data of training and testing
    '''
    num_feature = 0.98
    feature_num = 1280
    # source data
    sourceDomainPath_0 = "../data/source"
    sourceDomainCsv_0 = "../%s/%d/datainfo/source/source_raw_csv_0.txt" % (
        group_data_info, itemtime)
    source_raw = "../%s/%d/datainfo/source/source_raw_0.txt" % (group_data_info, itemtime)
    source_raw_label = "../%s/%d/datainfo/source/source_raw_label_0.txt" % (group_data_info, itemtime)
    source_csv = '../%s/%d/datainfo/source/source_csv_0.txt' % (group_data_info, itemtime)
    source_0 = "../%s/%d/datainfo/source/source_0.txt" % (group_data_info, itemtime)
    source_label_0_name = "../%s/%d/datainfo/source/source_label_0.txt" % (group_data_info, itemtime)
    source_0_raw_data, source_0, source_label_0 = create_pca_data(sourceDomainPath_0, sourceDomainCsv_0, feature_num,
                                                                  source_raw, source_raw_label, source_csv, source_0,
                                                                  source_label_0_name, num_feature, 7)
    # source data
    targetDomainPath = "../data/target"
    targetDomainCsv = "../%s/%d/datainfo/target/target_raw_csv.txt" % (
        group_data_info, itemtime)
    target_raw = "../%s/%d/datainfo/target/target_raw_1.txt" % (group_data_info, itemtime)
    target_raw_label = "../%s/%d/datainfo/target/target_raw_label_1.txt" % (group_data_info, itemtime)
    target_csv = '../%s/%d/datainfo/target/target_csv_1.txt' % (group_data_info, itemtime)
    target_name = "../%s/%d/datainfo/target/target_1.txt" % (group_data_info, itemtime)
    target_label_name = "../%s/%d/datainfo/target/target_label_1.txt" % (group_data_info, itemtime)
    target_raw_data, target, target_label = create_pca_data(targetDomainPath, targetDomainCsv, feature_num,
                                                            target_raw, target_raw_label, target_csv, target_name,
                                                            target_label_name, num_feature, 7)

    # gallery data
    gallery_set_path = "../data/gallery"
    gallery_csv = "../%s/%d/datainfo/gallery/gallery_csv.txt" % (group_data_info, itemtime)
    creat_csv.creat_csv(gallery_set_path, gallery_csv)
    gallery_, gallery_label = pictomat.imagetovec(gallery_csv, feature_num)
    gly_raw_flname = "../%s/%d/datainfo/gallery/gallery_raw.txt" % (group_data_info, itemtime)
    inputAndoutput.writefile(gallery_, gly_raw_flname)

    # probe data

    probe_set_path = "../data/probe"
    probe_raw_csv = "../%s/%d/datainfo/probe/probe_raw_csv.txt" % (group_data_info, itemtime)
    creat_csv.creat_csv(probe_set_path, probe_raw_csv)
    probe_raw, probe_raw_label = pictomat.imagetovec(probe_raw_csv, feature_num)
    prb_raw_flname = "../%s/%d/datainfo/probe/probe_raw.txt" % (group_data_info, itemtime)
    inputAndoutput.writefile(probe_raw, prb_raw_flname)
    prb_raw_label = "../%s/%d/datainfo/probe/probe_raw_label.txt" % (group_data_info, itemtime)
    inputAndoutput.writefile(probe_raw_label, prb_raw_label)

    probe_raw_csv_ = (open(probe_raw_csv))
    probe_csv_ = []
    for row in probe_raw_csv_:
        probe_csv_.append(row)
    probe_, probe_label, probe_csv = rand_sample(probe_raw, probe_raw_label, probe_csv_)
    prb_csv_flname = '../%s/%d/datainfo/probe/probe_csv.txt' % (group_data_info, itemtime)
    try:
        with open(prb_csv_flname, 'w') as fi:
            for row in probe_csv:
                row, iv = row.split('\n')
                fi.write(str(row) + '\n')
            fi.close()
    except IOError:
        prb_csv_path = os.path.dirname(prb_csv_flname)
        os.makedirs(prb_csv_path)
        with open(prb_csv_flname, 'w') as fi:
            for row in probe_csv:
                row, iv = row.split('\n')
                fi.write(str(row) + '\n')
            fi.close()

    prb_s_raw_flname = "../%s/%d/datainfo/probe/probe_.txt" % (group_data_info, itemtime)
    inputAndoutput.writefile(probe_, prb_s_raw_flname)

    # transfer data
    trans_gallery, trans_probe = skpca(target_raw_data, gallery_, probe_, k=num_feature)
    trans_gallery_flname = '../%s/%d/datainfo/gallery/trans_gallery.txt' % (group_data_info, itemtime)
    trans_probe_flname = '../%s/%d/datainfo/probe/trans_probe.txt' % (group_data_info, itemtime)
    inputAndoutput.writefile(trans_gallery, trans_gallery_flname)
    inputAndoutput.writefile(trans_probe, trans_probe_flname)
    gallery = np.array(trans_gallery)
    probe = np.array(trans_probe)

    # Save data
    source_0_flname = '../%s/%d/datainfo/source/Xs_0.txt' % (group_data_info, itemtime)
    source_label_0_flname = '../%s/%d/datainfo/source/Xs_label_0.txt' % (group_data_info, itemtime)
    target_flname = '../%s/%d/datainfo/target/Xt.txt' % (group_data_info, itemtime)
    target_label_flname = '../%s/%d/datainfo/target/Xt_label.txt' % (group_data_info, itemtime)
    gallery_flname = '../%s/%d/datainfo/gallery/gallery.txt' % (group_data_info, itemtime)
    gallery_label_flname = '../%s/%d/datainfo/gallery/gallery_label.txt' % (group_data_info, itemtime)
    probe_flname = '../%s/%d/datainfo/probe/probe.txt' % (group_data_info, itemtime)
    probe_label_flname = '../%s/%d/datainfo/probe/probe_label.txt' % (group_data_info, itemtime)

    inputAndoutput.writefile(source_0, source_0_flname)
    inputAndoutput.writefile(source_label_0, source_label_0_flname)
    inputAndoutput.writefile(target, target_flname)
    inputAndoutput.writefile(target_label, target_label_flname)
    inputAndoutput.writefile(gallery, gallery_flname)
    inputAndoutput.writefile(gallery_label, gallery_label_flname)
    inputAndoutput.writefile(probe, probe_flname)
    inputAndoutput.writefile(probe_label, probe_label_flname)

    return source_0, target, gallery, probe, source_label_0, \
           target_label, gallery_label, probe_label


def rand_sample(data, data_label, data_csv, num=4):
    """
    random select data from the input data
    :param data: input data ,numpy array
    :param data_label: the label of data
    :param data_csv:
    :param num: random seed ,type:int
    :return: select data
    """
    num_rand = num
    label_unique = np.unique(data_label)
    data_label.shape = (len(data_label))
    data_label_num_count = np.bincount(data_label)
    newData = np.zeros((data.shape[0], (len(label_unique) * num_rand)))
    ind = 0
    index = 0
    label = np.zeros((len(label_unique) * num_rand, 1), int)
    csv = []
    for i in range(len(label_unique)):
        try:
            rand = random.sample(range(data_label_num_count[i]), num_rand)
        except ValueError:
            num_rand = data_label_num_count[i]
            rand = random.sample(range(data_label_num_count[i]), num_rand)
        for j in range(num_rand):
            newData[:, ind] = data[:, rand[j] + index]
            label[ind, 0] = data_label[rand[j] + index]
            csv.append(data_csv[rand[j] + index])
            ind += 1
        index += data_label_num_count[i]
        num_rand = num
    return newData, label, csv
