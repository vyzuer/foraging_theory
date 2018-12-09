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

_DEBUG = True
__DEBUG = True
___DEBUG = False
_max_photos = 20


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

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
        print 'connecting to cassandra'
    else:
        client = MongoClient('172.29.35.126:27019')
        print 'connecting to slave4'

    # db = client.foraging_imgseg_fv
    db = client.foraging_img_info
    col_seg_fv = db[poi]


def _get_a_scores():
    spath = dump_base + '/micro_poi/scores.list'
    scores = np.loadtxt(spath, skiprows=0)

    return scores


def load_labels():
    # load the labels
    labels = None
    num_components = None

    poi_base_dir = dump_base + '/micro_poi/'
    labels_file = poi_base_dir + 'labels.list'

    if not os.path.exists(labels_file):
        print 'Error: micro poi labels not found. cannot continue.'
        exit(0)
    else:
        labels = np.loadtxt(labels_file, dtype='int')
        num_components = np.max(labels) + 1

    return labels, num_components

def get_data():

    # the database for poi
    dataset_path = base_dir + '/poi_info.pkl'

    # first check if data is present 
    assert os.path.exists(dataset_path)

    try:
        data = pd.read_pickle(dataset_path)
    except IOError:
        print 'database error'
        exit(0)

    return data

def _init(poi):
    # load global variables
    load_globals(poi)

    np.set_printoptions(precision=4, suppress=True)


def get_concepts(pid):
    doc = col_seg_fv.find_one({'_id': pid})
    assert doc is not None
    ids = np.fromiter(doc['seg_ids'], dtype=int)

    concept_map = np.zeros(n_clusters)
    concept_map[ids] = 1.

    return concept_map


def copy_image(pid, mpoi_id):
    dst = dump_base_nas + '/iconic_images/' + str(mpoi_id)
    if not os.path.exists(dst):
        os.makedirs(dst)

    img_src = base_dir + '/images_500/' + str(pid) + '.jpg'
    img_dst = dst + '/' + str(pid) + '.jpg'

    shutil.copyfile(img_src, img_dst)

def add_photo(pid, cur_concept_gain, mpoi_id):
    flag = False
    min_con = 10

    pid_con = get_concepts(pid)
    concept_gain = np.maximum(cur_concept_gain, pid_con)
    gain = np.sum(concept_gain) - np.sum(cur_concept_gain)

    if gain > min_con:
        flag = True
        cur_concept_gain = concept_gain

        # copy the iconic photo
        copy_image(pid, mpoi_id)

    return flag, cur_concept_gain

def master(clean=False):

    # create the storage location
    dpath = dump_base_nas + '/iconic_images/'
    if not os.path.exists(dpath):
        os.makedirs(dpath)

    data = get_data()

    labels, num_components = load_labels()

    assert data.shape[0] == labels.size

    # get the photos list
    photos_list = data.iloc[:,6].values
    # load the ascores of images
    a_scores = _get_a_scores()

    cur_concept_gain = np.zeros(n_clusters)

    for i in range(num_components):
        # ids for this component
        ids = labels == i
        photos = photos_list[ids]
        scores = a_scores[ids]

        num_photos = 0
        cur_concept_gain[:] = 0

        if photos.size > 0:

            sorted_idx = np.argsort(scores)[::-1]
            sorted_photos = photos[sorted_idx]

            for j in range(photos.size):
                pid = sorted_photos[j]

                iconic_photo, cur_concept_gain = add_photo(pid, cur_concept_gain, i)
                if iconic_photo:
                    num_photos += 1

                if num_photos > _max_photos:
                    break

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


