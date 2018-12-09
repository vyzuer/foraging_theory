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
import igraph as ig

_DEBUG = True
__DEBUG = True
___DEBUG = False


"""
output
------
'mpoi_network/photo_freq.info'
'mpoi_network/mpoi_time.info'
'mpoi_network/trip_time.info'
'mpoi_network/edges_time.info'
'mpoi_network/mpoi_edges.info'
'mpoi_network/start_mpoi.info'
'mpoi_network/end_mpoi.info'
'mpoi_network/.graph_data'
"""

# class datetime.timedelta([days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]])
# _max_time_diff = datetime.timedelta(0, 0, 0, 0, 0, 2)
_max_time_diff = datetime.timedelta(hours=1)
_INF = 1e5

dump_base = None
dump_base_nas = None
base_dir = None
graph_flag = None
col_seg_fv = None

# add the code package to the path
sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv

n_clusters = gv.__NUM_CLUSTERS

def load_globals(poi):
    global dump_base
    global dump_base_nas
    global base_dir
    global graph_flag
    global col_seg_fv

    base_dir = gv.__dataset_path + poi
    dump_base = gv.__base_dir + poi
    dump_base_nas = gv.__base_dir_nas + poi

    graph_flag = dump_base + '/mpoi_network/.graph_data'

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


def _init(poi):
    # load global variables
    load_globals(poi)

    np.set_printoptions(precision=4, suppress=True)

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

def get_concepts(pid):
    doc = col_seg_fv.find_one({'_id': pid})
    assert doc is not None
    ids = np.fromiter(doc['seg_ids'], dtype=int)

    # freq_count = np.bincount(ids, minlength=n_clusters)

    concept_map = np.zeros(n_clusters)
    concept_map[ids] = 1.
    # print concept_map

    return concept_map


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


def compute_gain(c_con_gain, c_gain, pid, score):

    concepts_map = get_concepts(pid)
    # concepts_gain = concepts_map*score
    concepts_gain = concepts_map

    c_con_gain = np.maximum(c_con_gain, concepts_gain)
    c_gain = np.sum(c_con_gain)

    return c_con_gain, c_gain


def train_model(data, mpoi_id):
    const_b = 60.0 # offset to avoid 0 in log

    X_gain = data[:,0]
    X = np.log(X_gain+const_b).reshape(-1,1)
    y = data[:,1]

    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X, y)
    const_a = regr.coef_[0]
    const_c = regr.intercept_

    # save the R2 score
    r2score = regr.score(X, y)

    
    if ___DEBUG:
        # The coefficients
        print('Coefficients: \n', regr.coef_)
        print('Intercept: \n', regr.intercept_)
        # The mean square error
        print("Residual sum of squares: %.2f"
              % np.mean((regr.predict(X) - y) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % regr.score(X, y))
    
    if __DEBUG:
        # Plot outputs
        f_gain_curve = dump_base_nas + '/micro_poi/gain_curve/'
        if not os.path.exists(f_gain_curve):
            os.makedirs(f_gain_curve)
        fname = f_gain_curve + str(mpoi_id) + '.png'
        # plt.scatter(np.exp(X)-const_b, y,  color='black')
        plt.scatter(X_gain, y, color='g')
        # sorted_x = np.sort(X, axis=0)
        # sorted_y = regr.predict(sorted_x)
        # plt.plot(np.exp(sorted_x[:,0])-const_b, sorted_y, color='blue', linewidth=3)
        xc = np.arange(7200)  # plot for 2 hours
        yc = const_a*np.log(xc+const_b) + const_c
        plt.plot(xc, yc, color='r', linewidth=3)
         
        plt.savefig(fname, dpi=80)
        plt.close()
        
    return [const_a, const_b, const_c], r2score

def perform_regression(mpoi_gain_samples, num_mpois):
    """
        we will convert the diminishing function into a linear function first
        we want to train 
        y = a - b*exp(-c*x) , here a = 1000 (maximum possible gain, bcz we have 1000 concepts)
        alternative equation is y = a*log(x+b) + c, here b = 15 seconds, time to capture first photo
        which is a diminishing returns function.
            """
    """
        for linear model we need to store the three variables
        1 - scaling :a
        2 - shift : b
        3 - intercept : c
    """

    model_params = np.zeros(shape=(num_mpois, 3))
    r2scores = np.zeros(num_mpois)

    # iterate over the mpoi list and train a regression model
    for idx, mpoi in enumerate(mpoi_gain_samples):
        data_points = np.array(mpoi)
        # print data_points
        # train the model and get back the coefficients
        if data_points.size > 0:
            model_params[idx,:], r2scores[idx] = train_model(data_points, idx)

    # dump the model parameters
    spath = dump_base + '/micro_poi/model_params.list'
    np.savetxt(spath, model_params, fmt='%.6f')

    spath = dump_base + '/micro_poi/r2scores.list'
    np.savetxt(spath, r2scores, fmt='%.6f')

    if __DEBUG:
        print 'mean r2 scores: ', np.mean(r2scores)



def get_graph(network_edges, num_nodes):

    nodes = np.arange(num_nodes)

    graph = ig.Graph.Weighted_Adjacency(network_edges.tolist())

    edges = graph.get_edgelist()

    # print edges
    # print graph.es['weight']

    return graph, nodes, edges


def find_shortest_paths(network_edges, num_components):

    mpoi_graph, nodes, edges = get_graph(network_edges, num_components)

    # find shortest path between all pair of nodes
    paths = mpoi_graph.shortest_paths(weights='weight')

    # print paths
    path_list = [[] for i in range(num_components)]

    for i, node in enumerate(mpoi_graph.vs):
        path_list[i] = mpoi_graph.get_shortest_paths(node, weights='weight')
        assert node.index == i

    return paths, path_list


def _dump_info(paths, path_list, m_gain_mpoi, m_stay_mpoi, m_reach_mpoi):
    dump_path = dump_base + '/micro_poi/mpoi_info/'

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    dpath = dump_path + 'shortest_path.pickle'
    joblib.dump(paths, dpath)

    dpath = dump_path + 'path_list.pickle'
    joblib.dump(path_list, dpath)

    dpath = dump_path + 'gain.pickle'
    joblib.dump(m_gain_mpoi, dpath)

    dpath = dump_path + 'stay.pickle'
    joblib.dump(m_stay_mpoi, dpath)

    dpath = dump_path + 'reach.pickle'
    joblib.dump(m_reach_mpoi, dpath)


def _analyze_location(data):

    if ___DEBUG:
        print data.iloc[1,:].values
    labels, num_components = load_labels()

    assert data.shape[0] == labels.size

    cap_date = data.iloc[:,1].values
    struct_date = [parser.parse(d) for d in cap_date]
    hour_list = [d.hour for d in struct_date]

    # get the user list
    users_list = data.iloc[:,8].values
    # get the photos list
    photos_list = data.iloc[:,6].values
    # load the ascores of images
    a_scores = _get_a_scores()
    # find unique users
    u_users, u_pos = np.unique(users_list, return_inverse=True)
    num_users = len(u_users)

    if __DEBUG:
        print 'number of users: ', num_users
        print 'number of mpoi: ', num_components

    # parameteres to compute
    # number of photos per mpoi per user
    photos_mpoi = np.zeros((num_users, num_components))
    # time spent at each mpoi by users
    time_mpoi = np.zeros((num_users, num_components), dtype=int)
    # average trip duration for users
    trip_time = np.zeros(num_users)
    # edge information for network formed by connecting mpoi
    mpoi_network = np.zeros((num_components, num_components), dtype=int)
    # time information for each of the edges
    edge_time_users = np.zeros((num_components, num_components))
    # mark start and end mpoi
    start_marker = np.zeros(num_components, dtype=int)
    end_marker = np.zeros(num_components, dtype=int)
    num_trips = 0

    # to store the list of gain points for each mpoi
    mpoi_gain_samples = [[] for i in range(num_components)]
    cur_concept_gain = np.zeros(n_clusters)
    # store the gain per mpoi per trip as a list to compute average later
    mpoi_gain = [[] for i in range(num_components)]
    # store the average stay time at each mpoi
    mpoi_stay_time = [[] for i in range(num_components)]
    # average time required to reach this mpoi
    mpoi_reach_time = [[] for i in range(num_components)]


    # iterate over each of the user
    for i, u in enumerate(u_users):
        u_trip_dur = []
        # idx stores the index to photos of this particular user
        idx = u_pos == i

        mpoi_id = labels[idx]
        time_list = np.array(struct_date)[idx]
        u_photos = photos_list[idx]
        u_scores = a_scores[idx]
        if ___DEBUG:
            print 'user photos: ', u_photos

        # find total photos per mpoi for this user
        pic_count = np.bincount(mpoi_id, minlength=num_components)
        photos_mpoi[i] = pic_count

        sorted_idx = np.argsort(time_list)

        sorted_mpoi = mpoi_id[sorted_idx]
        sorted_time = time_list[sorted_idx]
        sorted_photos = u_photos[sorted_idx]
        sorted_scores = u_scores[sorted_idx]
        # print sorted_mpoi
        # print sorted_time
        # print sorted_photos

        # iterate over each of the sorted photo and compute parameters
        t0 = sorted_time[0]
        trip_t0 = sorted_time[0]
        start_marker[sorted_mpoi[0]] += 1

        # find the gain for the first photo captured
        pid = sorted_photos[0]
        a_score = sorted_scores[0]

        cur_concept_gain = 0.0
        cur_gain = 0.0

        cur_concept_gain, cur_gain = \
            compute_gain(cur_concept_gain, cur_gain, pid, a_score)

        mpoi_gain_samples[sorted_mpoi[0]].append([0, cur_gain])

        for j in range(1, sorted_time.size):
            time_diff = sorted_time[j] - sorted_time[j-1]

            src_id = sorted_mpoi[j-1]
            dst_id = sorted_mpoi[j]
            # change of mpoi location
            if dst_id != src_id or time_diff > _max_time_diff:
                # if this is a different trip, update
                # relevant variables
                if time_diff > _max_time_diff:
                    trip_dur = sorted_time[j-1] - trip_t0
                    assert trip_dur.days == 0
                    if trip_dur.seconds > 0:
                        u_trip_dur.append(trip_dur.seconds)
                        num_trips += 1
                    trip_t0 = sorted_time[j]
                    # if trip duration is > 0 then mark the end mpoi
                    if trip_dur.seconds > 0.0:
                        end_marker[src_id] += 1
                        # mark the start mpoi as well
                        start_marker[dst_id] += 1
                else:
                    # if this is the same trip but just change in mpoi
                    # update the edge information
                    if time_diff.seconds > 0:
                        mpoi_network[src_id, dst_id] += 1
                        mpoi_network[dst_id, src_id] += 1

                    # also update the time information for this edge
                    assert time_diff.days == 0
                    if ___DEBUG:
                        print 'migration time: ', time_diff.seconds
                    if time_diff.seconds > 0:
                        edge_time_users[src_id, dst_id] += time_diff.seconds
                        edge_time_users[dst_id, src_id] += time_diff.seconds

                    # update the reach time list
                    if time_diff.seconds > 0:
                        mpoi_reach_time[dst_id].append(time_diff.seconds)

                stay_time = sorted_time[j-1] - t0
                if ___DEBUG:
                    print t0, sorted_time[j-1]
                    print 'stay time: ', stay_time

                # just for validation
                assert stay_time.days == 0
                
                time_mpoi[i][src_id] = max(stay_time.seconds, time_mpoi[i][src_id])

                if stay_time.seconds > 0:
                    mpoi_stay_time[src_id].append(stay_time.seconds)

                # update t0
                t0 = sorted_time[j]

                # update the current gain and concepts for a new mpoi
                # find the gain for the first photo captured
                pid = sorted_photos[j]
                a_score = sorted_scores[j]

                # add the gain from previous trip to mpoi list
                mpoi_gain[src_id].append(cur_gain)

                cur_concept_gain = 0.0
                cur_gain = 0.0

                cur_concept_gain, cur_gain = \
                    compute_gain(cur_concept_gain, cur_gain, pid, a_score)

                mpoi_gain_samples[sorted_mpoi[j]].append([0, cur_gain])
        
            else:
                # we are in the same mpoi
                pid = sorted_photos[j]
                a_score = sorted_scores[j]
        
                cur_concept_gain, cur_gain = \
                    compute_gain(cur_concept_gain, cur_gain, pid, a_score)

                t1 = sorted_time[j]
                if (t1-t0).seconds > 0:
                    # some photos have similar time stamp
                    mpoi_gain_samples[sorted_mpoi[j]].append([(t1 - t0).seconds, cur_gain])
        

        # add the gain from this trip to mpoi list
        mpoi_gain[sorted_mpoi[-1]].append(cur_gain)

        # update for the final photo
        stay_time = sorted_time[-1] - t0
        assert stay_time.days == 0
        time_mpoi[i][sorted_mpoi[-1]] = stay_time.seconds
        if ___DEBUG:
            print 'stay time: ', stay_time

        # update the trip duration
        trip_dur = sorted_time[-1] - trip_t0
        assert trip_dur.days == 0
        if trip_dur.seconds > 0:
            u_trip_dur.append(trip_dur.seconds)
            num_trips += 1
        user_trip_dur = np.array(u_trip_dur)
        if user_trip_dur.size > 0:
            m_tript = np.mean(user_trip_dur)
            trip_time[i] = m_tript

        # if trip duration is > 0 then mark the end mpoi
        if trip_dur.seconds > 0.0:
            end_marker[sorted_mpoi[-1]] += 1


    # take average of gain per mpoi
    m_gain_mpoi = np.zeros(num_components)
    for i, gain_list in enumerate(mpoi_gain):
        mgain = np.array(gain_list)
        if mgain.size > 0:
            m_gain_mpoi[i] = np.mean(mgain)


    if ___DEBUG:
        print m_gain_mpoi

    # compute average of stay time
    m_stay_mpoi = np.zeros(num_components)
    for i, s_time in enumerate(mpoi_stay_time):
        mstay = np.array(s_time)
        if mstay.size > 0:
            m_stay_mpoi[i] = np.mean(mstay)

    # update the mean reach time 
    m_reach_mpoi = np.zeros(num_components)
    for i, r_time in enumerate(mpoi_reach_time):
        mreach = np.array(r_time)
        if mreach.size > 0:
            m_reach_mpoi[i] = np.mean(mreach)

    if ___DEBUG:
        print m_reach_mpoi

    m_photos_mpoi = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 0, photos_mpoi)
    m_photos_mpoi[np.isnan(m_photos_mpoi)] = 0
    if ___DEBUG:
        print 'average num of photos per mpoi: \n', m_photos_mpoi

    m_time_mpoi = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 0, time_mpoi)
    m_time_mpoi[np.isnan(m_time_mpoi)] = 0
    if ___DEBUG:
        print 'average time per mpoi: \n', m_time_mpoi

    m_trip_time = np.mean(trip_time[np.nonzero(trip_time)])
    if __DEBUG:
        print 'average user trip duration: \n', m_trip_time

    m_edge_time_users = edge_time_users/mpoi_network
    m_edge_time_users[np.isnan(m_edge_time_users)] = np.max(m_edge_time_users[np.isfinite(m_edge_time_users)])
    for i in range(num_components):
        m_edge_time_users[i,i] = 0

    if ___DEBUG:
        print 'mpoi network: \n', mpoi_network
        # print 'cumulative edge time information: \n', edge_time_users
        print 'edge time information: \n', m_edge_time_users

    if ___DEBUG:
        print 'start mpoi list:\n', start_marker
        print 'end mpoi list:\n', end_marker

    if _DEBUG:
        print 'network analysis done.'

    # perform linear regression to fit diminishing gain for each mpoi
    if _DEBUG:
        print 'performing linear regression for diminishing gains...'

    # this will train an dump the linear models for all the mpois
    perform_regression(mpoi_gain_samples, num_components)

    if _DEBUG:
        print 'linear regression done..'

    # create graph and find shortest paths and pickle everything for later use
    if _DEBUG:
        print 'finding shortest paths...'

    paths, path_list = find_shortest_paths(m_edge_time_users, num_components)
    _dump_info(paths, path_list, m_gain_mpoi, m_stay_mpoi, m_reach_mpoi)

    if _DEBUG:
        print 'shortest path done..'

    return m_photos_mpoi, m_time_mpoi, m_trip_time, m_edge_time_users, mpoi_network, start_marker, end_marker, num_trips


def _sanity_check():
    dump_path = dump_base + '/mpoi_network/'

    spath = dump_path + 'photo_freq.info'
    if not os.path.exists(spath):
        return False

    spath = dump_path + 'mpoi_time.info'
    if not os.path.exists(spath):
        return False

    spath = dump_path + 'trip_time.info'
    if not os.path.exists(spath):
        return False

    spath = dump_path + 'edges_time.info'
    if not os.path.exists(spath):
        return False

    spath = dump_path + 'mpoi_edges.info'
    if not os.path.exists(spath):
        return False

    spath = dump_path + 'start_mpoi.info'
    if not os.path.exists(spath):
        return False

    spath = dump_path + 'end_mpoi.info'
    if not os.path.exists(spath):
        return False

    spath = dump_path + 'mpoi.info'
    if not os.path.exists(spath):
        return False

    return True

def _load_network_data():

    dump_path = dump_base + '/mpoi_network/'

    spath = dump_path + 'photo_freq.info'
    photo_freq = np.loadtxt(spath)

    spath = dump_path + 'mpoi_time.info'
    mpoi_time = np.loadtxt(spath)

    spath = dump_path + 'trip_time.info'
    trip_time = np.loadtxt(spath)

    spath = dump_path + 'edges_time.info'
    network_edge = np.loadtxt(spath)

    spath = dump_path + 'mpoi_edges.info'
    network = np.loadtxt(spath)

    spath = dump_path + 'start_mpoi.info'
    m_start = np.loadtxt(spath)

    spath = dump_path + 'end_mpoi.info'
    m_end = np.loadtxt(spath)

    return photo_freq, mpoi_time, trip_time, network_edge, network, m_start, m_end

def _dump_data(photo_freq, mpoi_time, trip_time, network_edge, network, m_start, m_end, num_trips):
    dump_path = dump_base + '/mpoi_network/'

    if not os.path.exists(dump_path):
        os.makedirs(dump_path)

    # photo frequency
    spath = dump_path + 'photo_freq.info'
    np.savetxt(spath, photo_freq, fmt='%.6f')

    spath = dump_path + 'mpoi_time.info'
    np.savetxt(spath, mpoi_time, fmt='%.6f')

    spath = dump_path + 'trip_time.info'
    np.savetxt(spath, np.array([trip_time]), fmt='%.6f')

    spath = dump_path + 'edges_time.info'
    np.savetxt(spath, network_edge, fmt='%.6f')

    spath = dump_path + 'mpoi_edges.info'
    np.savetxt(spath, network, fmt='%d')

    spath = dump_path + 'start_mpoi.info'
    np.savetxt(spath, m_start, fmt='%d')

    spath = dump_path + 'end_mpoi.info'
    np.savetxt(spath, m_end, fmt='%d')

    spath = dump_path + 'mpoi.info'
    np.savetxt(spath, np.array([num_trips]), fmt='%d')


def _debug_data():
    # load the data
    data = get_data()

    labels, num_components = load_labels()

    assert data.shape[0] == labels.size

    pos = data.iloc[:,{4,5}].values

    xmin = np.min(pos[:,0])
    xmax = np.max(pos[:,0])
    ymin = np.min(pos[:,1])
    ymax = np.max(pos[:,1])

    for i in range(num_components):
        ids = labels == i
        X = pos[ids]
        # Plot outputs
        f_mpoi = dump_base_nas + '/micro_poi/mpoi_plots/'
        if not os.path.exists(f_mpoi):
            os.makedirs(f_mpoi)
        fname = f_mpoi + str(i) + '.png'
        plt.scatter(X[:,0], X[:,1], color='g')
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(fname, dpi=80)
        plt.close()


def _perform_analysis():
    # load the data
    if _DEBUG:
        print 'loading data...'
    data = get_data()

    # analyze photo taking behaviour
    if _DEBUG:
        print 'starting network analysis...'

    photo_freq, mpoi_time, trip_time, network_edge, network, m_start, m_end, num_trips = _analyze_location(data)

    # dump the data for later use
    if _DEBUG:
        print 'dumping network analysis data...'
    _dump_data(photo_freq, mpoi_time, trip_time, network_edge, network, m_start, m_end, num_trips)

    open(graph_flag, 'a').close()

    return photo_freq, mpoi_time, trip_time, network_edge, network, m_start, m_end


def master(clean=False):
    photo_freq = None
    mpoi_time = None
    trip_time = None
    network_edge = None
    network = None

    # check if features already extracted
    if not clean and os.path.exists(graph_flag) and _sanity_check():
        if _DEBUG:
            print 'graph already constructed, loading data from disk...'
        photo_freq, mpoi_time, trip_time, network_edge, network, m_start, m_end = _load_network_data()
    else:
        if _DEBUG:
            print 'Performing mpoi analysis for graph structure...'
        photo_freq, mpoi_time, trip_time, network_edge, network, m_start, m_end = _perform_analysis()


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

    _debug_data()

