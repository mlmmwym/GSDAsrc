#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author by Elijah_Yi

from numpy import  *
from PIL import Image
from compiler.ast import flatten
import xlsxwriter
import numpy as np


def ImageToMatrix(filename):
    # load image
    im = Image.open(filename)
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='int16')
    return data

def MatrixToImage(data):
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def loadImage(filename):
    """
    load image
    :param filename: image path
    :return:
    """
    data_l = ImageToMatrix(filename)
    return data_l
def saveExcel(data,filename):
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            worksheet.write(i, j, data[i, j])
            j += 1
        i += 1
    workbook.close()

    voctor = flatten(data.tolist())
    print ("write success!")

def writefile(mat,filename):
    mat = np.array(mat,dtype=int16)
    with open (filename,'w') as fi:
        for i in range(mat.shape[0]):
            for j in range((mat.shape[1])):
                fi.write(str(mat[i][j])+'    ')
            fi.write('\n')
            i+=1
        fi.close()
    print ("%s Write success！"%filename)

def imagetovec(filename,sampleRol):
    rol = sampleRol#样本的长度与宽度的乘积
    file = open(filename)
    lines = len(file.readlines())
    dataMat = np.zeros([rol,lines],dtype='float16')#rol代表每个样本的维度 即图片的宽度乘以长度
    lable = np.zeros([lines,1],dtype="int16")#行向量每个元素代表对应维的类别
    i =0
    for line in open(filename):
        c, v = line.split(',')
        lable [i]= v
        data = ImageToMatrix(c)
        dataMat[:,i] = data
        i+=1
    return dataMat,lable
