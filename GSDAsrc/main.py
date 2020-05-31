#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author by Elijah_Yi
import os
import training
import test
import inputAndoutput
import FLD
import data_convert
import numpy as np
import optimizes_Vg
import warnings

warnings.filterwarnings("ignore")


# 两个gallery
def main(group_data_in, group_re, item, flag):
    '''
    :param group_data_in: the path of input data
    :param group_re: i-th time experiment of result
    :param item: i-th time experiment
    :param flag: 1 is to regenerate data, 0 is to run experiments from existing data
    :return: the accuracy of face verification
    '''
    itemtime = item
    group_data_info = group_data_in
    group_resu = group_re
    probe_csv = '../%s/%d/datainfo/probe/probe_csv.txt' % (group_data_info, itemtime)

    if flag == 0:
        # the path of data
        src_0_flname = '../%s/%d/datainfo/source/Xs_0.txt' % (group_data_info, itemtime)
        src_label_0_flname = '../%s/%d/datainfo/source/Xs_label_0.txt' % (group_data_info, itemtime)
        tar_flname = '../%s/%d/datainfo/target/Xt.txt' % (group_data_info, itemtime)
        gly_flname = '../%s/%d/datainfo/gallery/gallery.txt' % (group_data_info, itemtime)
        gly_label_flname = '../%s/%d/datainfo/gallery/gallery_label.txt' % (group_data_info, itemtime)
        prb_flname = '../%s/%d/datainfo/probe/probe.txt' % (group_data_info, itemtime)
        prb_label_flname = '../%s/%d/datainfo/probe/probe_label.txt' % (group_data_info, itemtime)

        source_0 = np.loadtxt(src_0_flname)
        source_label_0 = np.loadtxt(src_label_0_flname)
        target = np.loadtxt(tar_flname)
        gallery = np.loadtxt(gly_flname)
        gallery_label = np.loadtxt(gly_label_flname)
        probe = np.loadtxt(prb_flname)
        probe_label = np.loadtxt(prb_label_flname)

    else:
        source_0, target, gallery, probe, source_label_0, \
        _, gallery_label, probe_label = data_convert.creat_domain(group_data_info,
                                                                  itemtime)
    tar_gal = np.hstack((target, gallery))
    ws, wt, vs, vT = training.training(source_0, target, group_data_info, group_resu, itemtime)
    Xst = np.dot(target, vs)
    Xst = Xst.real
    # W_T * x_g
    p = np.dot(wt.T, gallery)
    # q_j can be considered as  the representative of the j-th subject in the source training
    # set in the common subspace,Q = [q1, q2, ..., qcs]
    Q = np.dot(ws.T, source_0)

    q_mean, q_n = FLD.statistics(Q, source_label_0)

    xst_mean, xst_n = FLD.statistics(Xst, source_label_0)
    # the coefficients for reconstructing  in the gallery in the common subspace.
    vg = optimizes_Vg.optimizeVg(p, q_mean)
    # the Intra-class scatter matrix of gallery
    sw_s = FLD.calc_Sw(Xst, source_label_0, xst_mean, xst_n, vg)
    st_g = FLD.within_class_S_T(gallery)
    # estimated inter class scatter matrix of gallery
    sb_g = st_g - sw_s
    # Projection matrix：Wfld_n_tf_sspp
    Wfld_n_tf_sspp = FLD.lda(Xst, tar_gal, sb_g, source_label_0, gallery_label)
    Wfld_n_flname = "../%s/%d/datainfo/parameter/Wfld_n_tfsspp_%d.txt" % (group_resu, itemtime, itemtime)
    inputAndoutput.writefile(Wfld_n_tf_sspp, Wfld_n_flname)

    acc_tf_sspp = test.test(gallery, probe, gallery_label, probe_label, Wfld_n_tf_sspp, probe_csv, group_resu, itemtime)
    print('the accury is :', acc_tf_sspp)

    return acc_tf_sspp


if __name__ == '__main__':

    flag = 0
    groups = ['all_domains']

    for group in groups:
        sum_ = 0.
        count = 0.
        groups_result = 'result/' + group
        for item in range(1, 11, 1):
            count += 1
            # print '\nthe %d times produce training sample'%item
            group_info = 'info/' + group
            acc = main(group_info, groups_result, item, flag)
            sum_ = sum_ + acc
        average = sum_ / count
        # print 'the average is :',average
        flname = '../' + groups_result + '.txt'
        try:
            wri = open(flname, 'w')
            wri.write(str(average))
            wri.close()
        except IOError:
            acc_fl_path = os.path.dirname(flname)
            os.makedirs(acc_fl_path)
            wri = open(flname, 'w')
            wri.write(str(average))
            wri.close()
