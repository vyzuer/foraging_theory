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
import shutil
import h5py
from scipy.stats import spearmanr
from scipy.stats import pearsonr

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


def load_data():
    dump_path = dump_base + '/micro_poi/mpoi_info/'

    assert os.path.exists(dump_path)

    dpath = dump_path + 'gain.pickle'
    gain = joblib.load(dpath)

    dpath = dump_path + 'stay.pickle'
    stay_time = joblib.load(dpath)

    spath = dump_base + '/micro_poi/mpoi_attractiveness.list'
    x = np.loadtxt(spath, skiprows=1)
    lpop = x[:,1]
    spop = x[:,2]

    spath = dump_base + '/mpoi_network/mpoi_time.info'
    mpoi_time = np.loadtxt(spath)


    return gain, stay_time, spop, lpop, mpoi_time

def _load_topic_distrib():
    spath = dump_base + '/mpoi_profiling/topic_distrib.h5'

    h5f = h5py.File(spath, 'r')
    topic_distrib = h5f['dataset_1'][:]
    h5f.close()

    return topic_distrib


def compute_simpsons_diversity(topic_distrib):
    num_components = topic_distrib.shape[0]

    div = np.zeros(num_components)

    for i in range(num_components):
        N = np.sum(topic_distrib[i])
        for t in topic_distrib[i]:
            div[i] += t*(t-1)

        div[i] = div[i]/(N*(N-1))

    return div

def compute_shanon_diversity(topic_distrib):
    num_components = topic_distrib.shape[0]

    div = np.zeros(num_components)

    for i in range(num_components):
        td = topic_distrib[i]/np.sum(topic_distrib[i])
        # td = topic_distrib[i]
        for t in td:
            if t > 0:
                div[i] += t*np.log(t)

    return -div

def compute_cor(x,y):
    ids = np.nonzero(y)
    x1 = x[ids]
    y1 = y[ids]
    c = spearmanr(x,y)

    return c

def master(poi, clean=False):

    gscore = 0.
    tscore = 0.
    sscore = 0.
    lscore = 0.
    t2score = 0.

    if _DEBUG:
        print 'initializing...'
    _init(poi)

    if _DEBUG:
        print 'starting master...'

    gain, stay_time, spop, lpop, mpoi_time = load_data()

    # load the topic distribution
    topic_distrib = _load_topic_distrib()

    # compute shanon diversity score
    div_score = compute_shanon_diversity(topic_distrib)
    # div_score = compute_simpsons_diversity(topic_distrib)
    print np.std(div_score)
    
    pi_mpois = np.array([g/s if s > 60 and g < 500 else 0 for g,s in zip(gain, stay_time)])

    # compute corelationm cofficients
    gscore = compute_cor(gain, div_score)
    tscore = compute_cor(stay_time, div_score)
    sscore = compute_cor(spop, div_score)
    lscore = compute_cor(pi_mpois, div_score)
    t2score = compute_cor(gain, stay_time)
    lpop_stay = compute_cor(lpop, stay_time)
    spop_stay = compute_cor(spop, stay_time)

    # print lpop_stay, spop_stay

    print gscore, tscore, sscore, lscore, t2score

    return [gscore, tscore, sscore, lscore, t2score, lpop_stay]

def invoke_master(poi_list, clean=False):
    global res_base_nas
    res_base_nas = gv.__res_dir_nas

    dir_name = res_base_nas + 'cor_measure/'
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)

    pois = np.loadtxt(poi_list, dtype=str)

    scores = []

    for poi in pois:
        print poi
        score = master(poi, clean=clean)
        scores.append(score)

    scores = np.mean(scores, axis=0)
    print scores


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_name"
        exit(0)

    poi_list = str(sys.argv[1])

    if ___DEBUG:
        print 'starting master...'

    invoke_master(poi_list, clean=True)

