#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author by Elijah_Yi
from numpy import *
import numpy as np
import os


def loadMat(filename,type='float64'):
    """
    load data
    :param filename: filename
    :param type:
    :return:
    """
    mat = np.loadtxt(filename,dtype=type)
    return  mat

def initalizeW(row,col):
    """
    initialize project matrix
    :param row: the number of row
    :param col: the number of column
    :return: project
    """
    inMat = np.random.randint(1,10,size = (row,col),dtype=int64)#初始化原始矩阵
    suMat = np.zeros((1,inMat.shape[1]),dtype=float64)
    for i in  range(inMat.shape[1]):
        suMat[0,i] = dot(inMat[:,i],inMat[:,i].T)
    W = np.zeros((row,col),dtype=float64)

    for i in range(inMat.shape[1]):
        for j in range(inMat.shape[0]):
            W[j,i] = float(inMat[j,i]/(np.sqrt(suMat[:,i])))
    return W

def writefile(mat,filename):
    """
    write mat to txt
    :param mat: data to write
    :param filename: the file to save
    :return:
    """
    mat = np.array(mat)
    try:
        with open(filename, 'w') as fi:
            for i in range(mat.shape[0]):
                for j in range((mat.shape[1])):
                    fi.write(str(mat[i][j]) + ' ')
                fi.write('\n')
                i += 1
            fi.close()
        print ("%s Write success！" % filename)
    except IOError:
        filePath = os.path.dirname(filename)
        os.makedirs(filePath)
        with open(filename, 'w') as fi:
            for i in range(mat.shape[0]):
                for j in range((mat.shape[1])):
                    fi.write(str(mat[i][j]) + ' ')
                fi.write('\n')
                i += 1
            fi.close()
        print ("%s Write success！" % filename)