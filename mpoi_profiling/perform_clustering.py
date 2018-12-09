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
import socket


"""
Output
---------
mpoi_profiling/random_samples.h5
mpoi_profiling/cluster_model/model.pkl
"""
sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

MAX_IMG_MPOI = gv.__MAX_IMG_MPOI
max_imgs = gv.__MAX_RAND_IMG

client = None
col_seg_fv = None
col_img_info = None

____DEBUG = False
___DEBUG = True
__DEBUG = True
_DEBUG = True

n_clusters = gv.__NUM_CLUSTERS

def _init(poi):
    global client
    global col_seg_fv
    global col_img_info

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
    else:
        client = MongoClient('172.29.35.126:27019')

    db = client.foraging_imgseg_fv
    db_img_info = client.foraging_img_info
    col_seg_fv = db[poi]
    col_img_info = db_img_info[poi]

def clustering(X):
    
    if _DEBUG:
        print 'performing clustering...'
    # Compute clustering with MiniBatchKMeans.
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=1000, max_iter=1000,
                          n_init=5, max_no_improvement=10, verbose=0)
    t0 = 0
    if __DEBUG:
        t0 = time()
    mbk.fit(X)
    t_mini_batch = time() - t0
    if __DEBUG:
        print("Time taken to run MiniBatchKMeans %0.2f seconds" % t_mini_batch)

    return mbk


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


def get_data(poi):

    # the database for poi
    dataset_path = gv.__dataset_path + poi + '/poi_info.pkl'

    img_list = None

    try:
        data = pd.read_pickle(dataset_path)
        # select only photo_id
        img_list = data.iloc[:, 6].astype(int).values
    except IOError:
        print 'database error'
        exit(0)

    return img_list


def _random_seletion(img_list, labels, num_components):
    if __DEBUG:
        print 'number of components: ', num_components

    if __DEBUG:
        print 'dataset size: ', img_list.size
    # iterate through all the components and select
    # only MAX_IMG_MPOI images per component
    total_imgs = 0
    # randomly select 100 images for clustering
    selection_matrix = np.zeros((num_components, max_imgs), dtype=bool)

    for i in range(num_components):
        imgs = img_list[labels == i]
        mpoi_size = imgs.size
        if ____DEBUG:
            print 'images per mpoi: ', mpoi_size

        # if num of images is more than MAX_IMG_MPOI 
        # then select MAX_IMG_MPOI images randomly
        if mpoi_size > MAX_IMG_MPOI:
            random_ids = random.sample(np.arange(min(mpoi_size, max_imgs)), MAX_IMG_MPOI)
            selection_matrix[i,random_ids] = True
            if __DEBUG:
                total_imgs += MAX_IMG_MPOI
        else:    
            selection_matrix[i,np.arange(mpoi_size)] = True
            if __DEBUG:
                total_imgs += mpoi_size

    if __DEBUG:
        print inspect.stack()[0][3], ': total selected images: ', total_imgs
        print inspect.stack()[0][3], ': total selected images: ', np.sum(selection_matrix)

    return selection_matrix


def get_random_samples(data, labels, num_components, sel_mat):
    X = []
    counter_list = np.zeros(num_components, dtype=int)
    comp_id_list = np.zeros(num_components, dtype=int)

    # iterate over the complete dataset and populate the data matrix
    items = col_seg_fv.find(no_cursor_timeout=True)

    assert items.count() == labels.size

    if __DEBUG:
        print inspect.stack()[0][3], ': total images: ', items.count()

    total_segs = 0
    total_imgs = 0
    missing_imgs = 0
    i = 0
    for doc in items:
        pid = doc['Photo_id']
        if ____DEBUG:
            print pid
        
        assert pid == data[i]

        comp_id = labels[i]

        # if not enough images for this component, 
        # check the current image for random selection
        if counter_list[comp_id] < MAX_IMG_MPOI:
            if sel_mat[comp_id, comp_id_list[comp_id]]:
                if ____DEBUG:
                    sel_mat[comp_id, comp_id_list[comp_id]] = False
                counter_list[comp_id] += 1

                # populate the data from mongodb
                n_segments = doc['n_segments']
                if n_segments > 0:
                    X.append(pickle.loads(doc['features']))

                    if __DEBUG:
                        total_segs += n_segments
                        total_imgs += 1
                else:
                    if ____DEBUG:
                        missing_imgs += 1

        comp_id_list[comp_id] += 1
        i += 1

    # concatenate all the matrices
    t0 = 0
    if __DEBUG:
        print 'forming data matrix...'
        t0 = time()
    X = np.concatenate(X, axis=0)
    if __DEBUG:
        print 'matrix formation time: ', time() - t0

    if __DEBUG:
        print inspect.stack()[0][3], ': total segments: ', total_segs
        print inspect.stack()[0][3], ': total images: ', total_imgs
        if ____DEBUG:
            print inspect.stack()[0][3], ': remaining images: ', np.sum(sel_mat)
            print inspect.stack()[0][3], ': missing images: ', missing_imgs
            print comp_id_list
            print counter_list

    items.close()

    if __DEBUG:
        print 'data size: ', X.shape

    return X


def dump_rand_samples(X, poi):
    spath = gv.__base_dir + poi + '/mpoi_profiling/'

    if not os.path.exists(spath):
        os.makedirs(spath)

    t0 = time()
    fname = spath + 'random_samples.h5'
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


def load_from_disk(poi):
    X = None

    spath = gv.__base_dir + poi + '/mpoi_profiling/random_samples.h5'

    if os.path.exists(spath):
        if _DEBUG:
            print 'loading random samples from disk...'
        h5f = h5py.File(spath, 'r')
        X = h5f['dataset_1'][:]
        h5f.close()
    
    return X

def load_data(poi):
    # if already sampled load from disk
    X = load_from_disk(poi)

    labels, num_components = load_labels(poi)

    if X is None:
    
        if _DEBUG:
            print 'sampling random data for clustering...'
        # load database from disk
        data = get_data(poi)

        # generate selection matrix for randomly selecting images
        sel_mat = _random_seletion(data, labels, num_components)

        # load only partial data for clustering
        X = get_random_samples(data, labels, num_components, sel_mat)

        # save to disk
        dump_rand_samples(X, poi)

    return X, labels

def load_cluster_model(poi):
    model = None

    spath = gv.__base_dir + poi + '/mpoi_profiling/cluster_model/model.pkl'

    if os.path.exists(spath):
        if _DEBUG:
            print 'loading cluster model from disk...'
        model = joblib.load(spath)
    
    return model


def dump_model(poi, model):
    spath = gv.__base_dir + poi + '/mpoi_profiling/cluster_model/'
    if not os.path.exists(spath):
        os.makedirs(spath)

    fname = spath + 'model.pkl'

    joblib.dump(model, fname)


def get_cluster_model(poi, data, clean=False):
    model = None
    # if present in disk load, otherwise perfor clustering
    if not clean:
        model = load_cluster_model(poi)

    if model is None:
        model = clustering(data)
        dump_model(poi, model)

    return model


def _predict_and_update(poi, model, labels):
    """
        iterate through all the images in this dataset
        predict the cluster class for each of the segment
        and update the database
    """

    data = get_data(poi)

    items = col_seg_fv.find(no_cursor_timeout=True)

    i = 0
    for doc in items:
        if ____DEBUG:
            print 'mpoi label: ', labels[i]
        ids = []
        if doc['n_segments'] > 0:
            x = pickle.loads(doc['features'])
            ids = model.predict(x).tolist()

        col_seg_fv.update_one({'_id': doc['_id']}, {'$set':{'seg_ids': ids, 'mpoi_label': labels[i]}})

        assert doc['Photo_id'] == data[i]

        # need to update the new database as well
        new_doc = col_img_info.find_one({'_id':doc['Photo_id']})
        col_img_info.update_one({'_id': new_doc['_id']}, {'$set':{'seg_ids': ids, 'mpoi_label': labels[i]}})
        # doc['seg_ids'] = ids
        # doc['mpoi_label'] = labels[i]
        # update the database
        # col_seg_fv.save(doc)

        if ____DEBUG:
            print 'updated: ', doc['Photo_id']

        i += 1

    items.close()
    
def master(poi, clean=False):
    # load the data
    if _DEBUG:
        print 'loading data...'
    data, labels = load_data(poi)

    # get the cluster model 
    if _DEBUG:
        print 'getting cluster model...'
    model = get_cluster_model(poi, data, clean)

    # predict cluster for each of the segment
    # and update the mongodb database
    if _DEBUG:
        print 'predicting and updating cluster ids...'
    _predict_and_update(poi, model, labels)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_name"

    poi = str(sys.argv[1])

    _init(poi)

    master(poi, clean=True)

