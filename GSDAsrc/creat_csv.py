#!/usr/bin/env python  

import sys
import os.path


# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

def creat_csv(basepath, filename):
    BASE_PATH = basepath
    SEPARATOR = ","
    try:
        fh = open(filename, 'w')
    except IOError:
        flpath = os.path.dirname(filename)
        os.makedirs(flpath)
        fh = open(filename, 'w')
    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s/%s" % (subject_path, filename)
                # print "%s%s%d" % (abs_path, SEPARATOR, label)
                fh.write(abs_path)
                fh.write(SEPARATOR)
                fh.write(str(label))
                fh.write("\n")
            label = label + 1
    fh.close()
