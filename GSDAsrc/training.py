#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author by Elijah_Yi

import os
import datetime
import numpy as np
import numpy.linalg as la
import optimizesV
import optimizesW
import inputAndoutput


def training(source, target, group_data_infoma, group_resu, itemtime):
    """
    :param source: source data
    :param target: target data
    :param group_data_infoma: input data i-th experiment
    :param group_resu: i-th experiment result
    :param itemtime: i-th experiment
    :return: project matrix
    """
    col = 300
    group_result = group_resu
    print('source.shape:', source.shape)
    print('target.shape:', target.shape)
    iterations = 0
    max_iterations = 200
    epsilon = 1000

    # Randomly initialize the projection matrix
    Ws = inputAndoutput.initalizeW(source.shape[0], col)
    Wt = inputAndoutput.initalizeW(target.shape[0], col)

    Ws_flname = '../%s/%d/datainfo/parameter/ws_0_init.txt' % (group_result, itemtime)
    Wt_flname = '../%s/%d/datainfo/parameter/wt_0_init.txt' % (group_result, itemtime)
    inputAndoutput.writefile(Ws, Ws_flname)
    inputAndoutput.writefile(Wt, Wt_flname)

    Vs_0 = optimizesV.optimizeV(source, target, Ws, Wt)
    Vt_0 = optimizesV.optimizeV(target, source, Wt, Ws)

    trace_name = '../%s/%d/datainfo/parameter/Trace_%d.txt' % (group_result, itemtime, itemtime)
    trace_a_name = '../%s/%d/datainfo/parameter/Trace_a_%d.txt' % (group_result, itemtime, itemtime)
    source_trace_name = '../%s/%d/datainfo/parameter/source_trace_%d.txt' % (group_result, itemtime, itemtime)
    target_trace_name = '../%s/%d/datainfo/parameter/target_trace_%d.txt' % (group_result, itemtime, itemtime)

    while (epsilon > 0.01 and iterations < max_iterations):
        iterations += 1
        # optimize project matrix
        Ws, Wt, T_0, Trace_a = optimizesW.optimizesW(source, target, Vs_0, Vt_0)
        max_var_source = np.dot(np.dot(Ws.T, source), np.dot(source.T, Ws))
        max_var_target = np.dot(np.dot(Wt.T, target), np.dot(target.T, Wt))
        max_trace_source = 0
        max_trace_target = 0

        for i in range(max_var_source.shape[0]):
            max_trace_source += max_var_source[i, i]

        for i in range(max_var_target.shape[0]):
            max_trace_target += max_var_target[i, i]

        trace_write(source_trace_name, max_trace_source, iterations)
        trace_write(target_trace_name, max_trace_target, iterations)

        # write Trace
        trace_write(trace_name, T_0, iterations)
        # write trace_a
        trace_write(trace_a_name, Trace_a, iterations)

        print("training time:", iterations, "\tepsilon:", epsilon, '\tTrace:', T_0)
        # optimize the coefficients of domian
        Vs_1 = optimizesV.optimizeV(source, target, Ws, Wt)
        Vt_1 = optimizesV.optimizeV(target, source, Wt, Ws)
        iterations += 1

        Ws, Wt, T_1, Trace_a = optimizesW.optimizesW(source, target, Vs_1, Vt_1)

        max_var_source = np.dot(np.dot(Ws.T, source), np.dot(source.T, Ws))
        max_var_target = np.dot(np.dot(Wt.T, target), np.dot(target.T, Wt))

        max_trace_source = 0
        max_trace_target = 0

        for i in range(max_var_source.shape[0]):
            max_trace_source += max_var_source[i, i]

        for i in range(max_var_target.shape[0]):
            max_trace_target += max_var_target[i, i]

        trace_write(source_trace_name, max_trace_source, iterations)
        trace_write(target_trace_name, max_trace_target, iterations)

        # write Trace
        trace_write(trace_name, T_1, iterations)
        # write trace_a
        trace_write(trace_a_name, Trace_a, iterations)

        print("training time:", iterations, "\tepsilon:", epsilon, '\tTrace:', T_1)
        # 优化 V
        Vs_0 = optimizesV.optimizeV(source, target, Ws, Wt)
        Vt_0 = optimizesV.optimizeV(target, source, Wt, Ws)

        epsilon = abs(T_1 - T_0)

    # write the coefficient and project matrix
    ws_0_flname = '../%s/%d/datainfo/parameter/Ws_%d.txt' % (group_result, itemtime, itemtime)
    wt_0_flname = '../%s/%d/datainfo/parameter/Wt_%d.txt' % (group_result, itemtime, itemtime)
    Vs_0_flname = '../%s/%d/datainfo/parameter/Vs_0_%d.txt' % (group_result, itemtime, itemtime)
    Vt_0_flname = '../%s/%d/datainfo/parameter/Vt_0_%d.txt' % (group_result, itemtime, itemtime)
    inputAndoutput.writefile(Ws, ws_0_flname)
    inputAndoutput.writefile(Wt, wt_0_flname)
    inputAndoutput.writefile(Vs_0, Vs_0_flname)
    inputAndoutput.writefile(Vt_0, Vt_0_flname)
    return Ws, Wt, Vs_0, Vt_0


def trace_write(flname_path, data, item):
    """
    write trace
    :param flname_path: the filename of save data
    :param data: the information to write
    :param item: i-th experiment
    :return:
    """
    try:
        trace_f = open(flname_path, 'a')
        trace_f.write(str(item) + ',' + str(data) + '\n')
        trace_f.close()
    except IOError:
        trace_a_path = os.path.dirname(flname_path)
        os.makedirs(trace_a_path)
        trace_f = open(flname_path, 'a')
        trace_f.write(str(item) + ',' + str(data) + '\n')
        trace_f.close()
