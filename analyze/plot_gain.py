from pymongo import MongoClient
import numpy as np
import sys
import pandas as pd
import os
from pprint import pprint
import datetime
from dateutil import parser
import subprocess
import time
import cv2
import collections as coll
import numpy as np
import time
import argparse
import threading
import bson.binary as bbin
from sklearn.externals import joblib
import socket
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib as mpl
import igraph as ig

_DEBUG = True
__DEBUG = True
___DEBUG = False



# class datetime.timedelta([days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]])
# _max_time_diff = datetime.timedelta(0, 0, 0, 0, 0, 2)
_max_time_diff = datetime.timedelta(hours=1)

dump_base = None
dump_base_nas = None
res_base_nas = None
base_dir = None
graph_flag = None
col_seg_fv = None
poi_list = None

# add the code package to the path
sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv

n_clusters = gv.__NUM_CLUSTERS

def load_globals(pois):
    global dump_base
    global dump_base_nas
    global res_base_nas
    global base_dir
    global graph_flag
    global col_seg_fv
    global poi_list

    base_dir = gv.__dataset_path
    dump_base = gv.__base_dir
    dump_base_nas = gv.__base_dir_nas
    res_base_nas = gv.__res_dir_nas 
    poi_list = pois


def _init(poi_list):
    # load global variables
    load_globals(poi_list)

    np.set_printoptions(precision=4, suppress=True)


def load_model_params(poi):
    dump_path = dump_base + poi
    spath = dump_path + '/micro_poi/model_params.list'
    params = np.loadtxt(spath)

    # ignore the params with 0 r2 score
    spath = dump_path + '/micro_poi/r2scores.list'
    r2scores = np.loadtxt(spath)

    ids = np.nonzero(r2scores)
    params = params[ids,:]

    return params[0,:,:]

def plot_average_gain():

    pois = np.loadtxt(poi_list, dtype=str)
    poi_label = np.loadtxt('../data/poi.label', dtype=str)

    fname = res_base_nas + '/gain_data.list'
    fp = open(fname, 'w')

    for poi,label in zip(pois, poi_label):
        print poi
        if poi is 'fhill':
            continue
        # load the models
        params = load_model_params(poi)

        X = []

        for param in params:
            [a, b, c] = param
            # ignore gain curves with negative slope and slope too high
            if a > 0 and a*np.log(4000)+c < 1000:
                x = np.arange(1,4000)
                y = a*np.log(x)+c
                X.append(y)

        # find the mean and standard deviation
        X = np.array(X)
        X_mean = np.mean(X, axis=0)

        np.savetxt(fp, X_mean.reshape(1,-1), fmt='%.6f', delimiter='\t')

        x = np.arange(1,4000)
        mpl.rcParams.update({'font.size': 18})
        plt.plot(x, X_mean, linewidth=3, label=label)
        plt.xlim(-10,4000)
        plt.ylim(-10,450)
        plt.xticks(np.arange(0,4001,1000))
        plt.yticks(np.arange(0,451,200))

    plt.xlabel('time')
    plt.ylabel('gain')
    plt.legend(loc=3, prop={'size':6})
    plt.tight_layout()

    res_base = res_base_nas + '/gain_curve/'
    if not os.path.exists(res_base):
        os.makedirs(res_base)
    fname = res_base + 'gain.png'
    plt.savefig(fname)
    plt.close()


def master(clean=False):

    #plot the average gain curve
    plot_average_gain()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_list"
        exit(0)

    poi_list = str(sys.argv[1])

    if _DEBUG:
        print 'initializing...'
    _init(poi_list)

    if _DEBUG:
        print 'starting master...'

    master(clean=True)


