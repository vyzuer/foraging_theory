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
poi_name = None

# add the code package to the path
sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv

n_clusters = gv.__NUM_CLUSTERS

def load_globals(poi):
    global dump_base
    global dump_base_nas
    global res_base_nas
    global base_dir
    global graph_flag
    global col_seg_fv
    global poi_name

    poi_name = poi
    base_dir = gv.__dataset_path + poi
    dump_base = gv.__base_dir + poi
    dump_base_nas = gv.__base_dir_nas + poi
    res_base_nas = gv.__res_dir_nas 


def _init(poi):
    # load global variables
    load_globals(poi)

    np.set_printoptions(precision=4, suppress=True)


def load_model_params():
    spath = dump_base + '/micro_poi/model_params.list'
    params = np.loadtxt(spath)

    # ignore the params with 0 r2 score
    spath = dump_base + '/micro_poi/r2scores.list'
    r2scores = np.loadtxt(spath)

    ids = np.nonzero(r2scores)
    params = params[ids,:]

    return params[0,:,:]

def plot_average_gain():
    # load the models
    params = load_model_params()

    X = []
    mean_params = []

    for param in params:
        [a, b, c] = param
        # ignore gain curves with negative slope and slope too high
        if a > 0 and a*np.log(4000)+c < 1000:
            x = np.arange(1,4000)
            y = a*np.log(x)+c
            X.append(y)
            mean_params.append([a,b,c])

    # find the mean and standard deviation
    X = np.array(X)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)

    res_base = res_base_nas + '/gain_curve/'
    if not os.path.exists(res_base):
        os.makedirs(res_base)
    fname = res_base + poi_name + '_gain.png'
    x = np.arange(1,4000)
    mpl.rcParams.update({'font.size': 30})
    plt.plot(x, X_mean, linewidth=5)
    plt.fill_between(x, X_mean-X_std, X_mean+X_std, facecolor='blue', alpha=0.2)
    plt.xlim(-10,4000)
    plt.ylim(-10,450)
    plt.xticks(np.arange(0,4001,1000))
    plt.yticks(np.arange(0,451,200))
    plt.savefig(fname)
    plt.close()

    # dump the average gain cure
    spath = dump_base + '/micro_poi/mean_params.list'
    mean_params = np.mean(np.array(mean_params), axis=0)
    np.savetxt(spath, mean_params.reshape(1,-1))

def r2scores():
    spath = dump_base + '/micro_poi/r2scores.list'
    scores = np.loadtxt(spath)

    print 'Mean R2 Score', np.mean(scores[np.nonzero(scores)])

def master(clean=False):

    # compute average r2 score for gain curves fitting
    r2scores()

    #plot the average gain curve
    plot_average_gain()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_name"

    poi = str(sys.argv[1])

    if _DEBUG:
        print 'initializing...'
    _init(poi)

    if _DEBUG:
        print 'starting master...'

    master(clean=True)


