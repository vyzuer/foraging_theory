from time import time
import numpy as np
import sys
import pandas as pd
import os
import random
from pymongo import MongoClient
import pickle
import inspect
import h5py
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation

"""
Output
------
mpoi_profiling/tf_data.h5
mpoi_profiling/topic_distrib.h5
"""

sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv
import socket

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

MAX_IMG_MPOI = gv.__MAX_IMG_MPOI
max_imgs = gv.__MAX_RAND_IMG

client = None
col_seg_fv = None

___DEBUG = False
__DEBUG = True
_DEBUG = True

n_clusters = gv.__NUM_CLUSTERS
n_topics = gv.__NUM_TOPICS
n_top_words = 20

def _init(poi):
    global client
    global col_seg_fv

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
        print 'connecting to cassandra'
    else:
        client = MongoClient('172.29.35.126:27019')
        print 'connecting to slave4'

    db = client.foraging_imgseg_fv
    col_seg_fv = db[poi]


def load_labels(poi):
    # load the labels
    labels = None
    num_components = None

    poi_base_dir = gv.__base_dir + poi + '/micro_poi/'
    labels_file = poi_base_dir + 'labels.list'

    if not os.path.exists(labels_file):
        print 'Error: micro poi labels not found. cannot continue.'
        exit(0)
    else:
        labels = np.loadtxt(labels_file, dtype='int')
        num_components = np.max(labels) + 1

    return labels, num_components


def _dump_data_samples(X, poi):
    spath = gv.__base_dir + poi + '/mpoi_profiling/'

    if not os.path.exists(spath):
        os.makedirs(spath)

    t0 = time()
    fname = spath + 'tf_data.h5'
    h5f = h5py.File(fname, 'w')
    h5f.create_dataset('dataset_1', data=X)
    h5f.close()
    print 'dump time: ', time() - t0

    """
    t0 = time()
    # load for testing
    h5f = h5py.File(fname,'r')
    b = h5f['dataset_1'][:]
    h5f.close()
    print 'load time: ', time() - t0

    """


def _load_from_disk(poi):
    X = None

    spath = gv.__base_dir + poi + '/mpoi_profiling/tf_data.h5'

    if os.path.exists(spath):
        if _DEBUG:
            print 'loading data samples from disk...'
        h5f = h5py.File(spath, 'r')
        X = h5f['dataset_1'][:]
        h5f.close()
    
    return X


def _dump_topic_distrib(poi, topic_distrib):
    spath = gv.__base_dir + poi + '/mpoi_profiling/topic_distrib.h5'

    h5f = h5py.File(spath, 'w')
    h5f.create_dataset('dataset_1', data=topic_distrib)
    h5f.close()


def _dump_model(poi, model):
    spath = gv.__base_dir + poi + '/mpoi_profiling/lda_model/'
    if not os.path.exists(spath):
        os.makedirs(spath)

    fname = spath + 'model.pkl'

    joblib.dump(model, fname)


def _get_tf_data(poi, clean=False):
    """
        form a tf matrix to perform lda topic modeling
    """

    # if data is formed earlier load from disk
    data = _load_from_disk(poi)

    if data is None or clean == True:
        labels, num_components = load_labels(poi)
        data = np.zeros((num_components, n_clusters), dtype=int)

        items = col_seg_fv.find(no_cursor_timeout=True)

        assert len(labels) == items.count()

        i = 0
        for doc in items:
            if ___DEBUG:
                print 'processing: ', doc['Photo_id']
            ids = np.fromiter(doc['seg_ids'], dtype=int)
            label = doc['mpoi_label']
            assert label == labels[i]
            
            # update data
            freq_count = np.bincount(ids, minlength=n_clusters)
            data[label, :] += freq_count

            if ___DEBUG:
                print 'processed: ', doc['Photo_id']

            i += 1

        items.close()

        _dump_data_samples(data, poi)

    return data
 
def print_top_words(model, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print topic.argsort()[:-n_top_words - 1:-1]

def _perform_topic_modeling(data):

    if __DEBUG:
        n_samples, n_features = data.shape
        print("Fitting LDA models with tf features")
        print("n_samples=%d and n_features=%d..." % (n_samples, n_features))

    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=1000,
                                    learning_method='online', learning_offset=50.,
                                    random_state=0)
    t0 = time()
    topic_distrib = lda.fit_transform(data)
    print("done in %0.3fs." % (time() - t0))

    # print_top_words(lda, n_top_words)
    # np.set_printoptions(precision=4, suppress=True)
    # print x[:,:]

    return lda, topic_distrib

def master(poi, clean=False):

    if _DEBUG:
        print 'starting topic modeling...'
    # load the data
    if _DEBUG:
        print 'loading data...'
    data = _get_tf_data(poi, clean=clean)

    # perform topic modeling
    if _DEBUG:
        print 'performing topic modeling...'
    model, topic_distrib = _perform_topic_modeling(data)

    # dump the lda model
    if _DEBUG:
        print 'dumping topic-model...'
    _dump_model(poi, model)

    if _DEBUG:
        print 'dumping topic distribution...'
    _dump_topic_distrib(poi, topic_distrib)

    if _DEBUG:
        print 'topic modeling done.'

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_name"

    poi = str(sys.argv[1])

    _init(poi)

    master(poi, clean=True)

