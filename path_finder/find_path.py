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
import tsp
# import plot_network
from collections import deque
import h5py
from sklearn.metrics.pairwise import cosine_similarity

_DEBUG = True
__DEBUG = False
___DEBUG = False

_INF = 1e5
_MIN_DIST = 2e-4

"""
output
------
"""

# class datetime.timedelta([days[, seconds[, microseconds[, milliseconds[, minutes[, hours[, weeks]]]]]]])
# _max_time_diff = datetime.timedelta(0, 0, 0, 0, 0, 2)
_max_time_diff = datetime.timedelta(hours=1)

dump_base = None
dump_base_nas = None
base_dir = None
col_seg_fv = None
scaler = None
_poi = None
attract = None
spop = None
lpop = None
mpoi_time = None
t_start_time = None
t_trip_dur = None

# weights for profitability
delta = 1.0
kappa = 1.0
theta = 1.0
eta = 1.0

# add the code package to the path
sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv
from common.Trip import Trip

n_clusters = gv.__NUM_CLUSTERS

class method:
    proposed, random, social, local, socloc, profit, psoc, ploc, personal  = range(9)

def load_globals(poi):
    global dump_base
    global dump_base_nas
    global base_dir
    global _poi
    global col_seg_fv

    _poi = poi

    base_dir = gv.__dataset_path + poi
    dump_base = gv.__base_dir + poi
    dump_base_nas = gv.__base_dir_nas + poi

    host_name = socket.gethostname()
    if host_name == 'cassandra':
        client = MongoClient()
        if __DEBUG:
            print 'connecting to cassandra'
    else:
        client = MongoClient('172.29.35.126:27019')
        if __DEBUG:
            print 'connecting to slave4'

    # db = client.foraging_imgseg_fv
    db = client.foraging_img_info
    col_seg_fv = db[poi]


def _init(poi):
    global attract
    global spop
    global lpop
    global mpoi_time

    # load global variables
    load_globals(poi)

    np.set_printoptions(precision=6, suppress=True)

    attract, spop, lpop, mpoi_time = _get_mpoi_qualities()


def load_data():
    """
        this function will load 
        - shortest path/list
        - gain
        - gain function
        - stay time
        - reach time
    """

    dump_path = dump_base + '/micro_poi/mpoi_info/'

    assert os.path.exists(dump_path)

    dpath = dump_path + 'shortest_path.pickle'
    paths = joblib.load(dpath)

    dpath = dump_path + 'path_list.pickle'
    path_list = joblib.load(dpath)

    dpath = dump_path + 'gain.pickle'
    gain = joblib.load(dpath)

    dpath = dump_path + 'stay.pickle'
    stay_time = joblib.load(dpath)

    dpath = dump_path + 'reach.pickle'
    reach_time = joblib.load(dpath)

    spath = dump_base + '/micro_poi/model_params.list'
    model_params = np.loadtxt(spath)

    return np.array(paths), path_list, gain, stay_time, reach_time, model_params

def _load_average_trip_duration():
    spath = dump_base + '/mpoi_network/trip_time.info'
    trip_time = np.loadtxt(spath)

    return trip_time/3600.


def _load_gmm():
    global scaler
    gmm_path = dump_base + '/micro_poi/gmm/model/gmm.pkl'

    if not os.path.exists(gmm_path):
        print 'Error: GMM model not found. exiting...'
        exit(0)

    gmm = joblib.load(gmm_path)

    scaler_path = dump_base + '/micro_poi/gmm/scaler/scaler.pkl'

    if not os.path.exists(scaler_path):
        print 'Error: Scaler model not found. exiting...'
        exit(0)

    scaler = joblib.load(scaler_path)

    return gmm, scaler


def _update_nodes_position(graph, gmm, scaler):

    num_nodes = graph.vcount()

    nodes_pos = gmm.means_

    if __DEBUG:
        print 'number of nodes in network: ', num_nodes

    # this happened for wasmon, reason was
    # the last component was empty
    if nodes_pos.shape[0] != num_nodes:
        nodes_pos = nodes_pos[:num_nodes,:]

    # normalized positions for distance calculation
    graph.vs['norm_position'] = nodes_pos

    nodes_pos = scaler.inverse_transform(nodes_pos)

    graph.vs['position'] = nodes_pos

    return graph


def _update_edges_position(graph):
    num_edges = graph.ecount()
    
    edges_pos = np.zeros((num_edges, 2, 3))
    # iterate over each of the edges and populate position
    for i, edge in enumerate(graph.es):
        assert edge.index == i
        edges_pos[i,0,:] = graph.vs[edge.source]['position']
        edges_pos[i,1,:] = graph.vs[edge.target]['position']

    graph.es['position'] = edges_pos

    return graph


def _load_graph():
    s_edges = dump_base + '/mpoi_network/edges_time.info'

    assert os.path.exists(s_edges)

    edges = np.loadtxt(s_edges)

    graph = ig.Graph.Weighted_Adjacency(edges.tolist())

    return graph


def _load_network():
    # load the gmm and scaler model for mpoi
    if __DEBUG:
        print 'loading gmm model from dump...'
    gmm, scaler = _load_gmm()

    # load the network information
    if __DEBUG:
        print 'loading graph info from dump...'
    graph = _load_graph()

    # get the nodes position from gmm model
    if __DEBUG:
        print 'getting nodes position...'
    graph = _update_nodes_position(graph, gmm, scaler)

    # get the edges position using the edges information
    if __DEBUG:
        print 'getting edges position...'
    graph = _update_edges_position(graph)

    return graph


def _distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

def update_node_dist(graph):
    n_nodes = graph.vcount()
    new_paths = np.zeros(shape=(n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            node1 = graph.vs[i]
            node2 = graph.vs[j]

            assert node1.index == i
            assert node2.index == j

            new_paths[i, j] = _distance(node1['norm_position'], node2['norm_position'])
            # new_paths[i, j] = _distance(node1['norm_position'][:2], node2['norm_position'][:2])
            # new_paths[i, j] = _distance(node1['position'][:2], node2['position'][:2])
            # new_paths[i, j] = _distance(node1['position'], node2['position'])
            new_paths[j, i] = new_paths[i, j]

    return new_paths/np.max(new_paths)

def find_end_node(graph, end_pos, start_time, trip_dur, start_node):
    end_time = start_time.hour + start_time.minute/60. + trip_dur.seconds/3600.
    # print s_time, start_time.hour, start_time.minute

    e_pos = np.array([end_pos[0], end_pos[1], end_time])
    e_pos = scaler.transform(e_pos.reshape(1,-1))

    min_dist = np.inf
    end_node = None
    for node in graph.vs:
        dist = _distance(e_pos, node['norm_position'])
        if dist < min_dist:
            min_dist = dist
            end_node = node.index

    return end_node


def find_start_node(graph, start_pos, start_time):
    s_time = start_time.hour + start_time.minute/60.
    # print s_time, start_time.hour, start_time.minute

    s_pos = np.array([start_pos[0], start_pos[1], s_time])
    s_pos = scaler.transform(s_pos.reshape(1,-1))

    min_dist = np.inf
    start_node = None
    for node in graph.vs:
        dist = _distance(s_pos, node['norm_position'])
        if dist < min_dist:
            min_dist = dist
            start_node = node.index

    return start_node

def select_start_end_node(graph, start_time, trip_dur, start_pos, end_pos):

    start_node = find_start_node(graph, start_pos, start_time)

    end_node = find_end_node(graph, end_pos, start_time, trip_dur, start_node)

    return start_node, end_node
    # return start_node, start_node


def _init_end_position(graph):
    # load the end node from disk
    spath = dump_base + '/mpoi_network/end_mpoi.info'
    end_nodes = np.loadtxt(spath, dtype=int)

    end_node = np.argmax(end_nodes)

    end_pos = graph.vs[end_node]['position']

    return end_pos[0], end_pos[1]


def _init_start_position(graph):
    # load the start and end node from disk
    spath = dump_base + '/mpoi_network/start_mpoi.info'
    start_nodes = np.loadtxt(spath, dtype=int)

    start_node = np.argmax(start_nodes)

    start_pos = graph.vs[start_node]['position']

    return start_pos[0], start_pos[1]


def _init_opt(graph, start_time, trip_dur, start_pos, end_pos):
    global t_start_time
    global t_trip_dur

    if start_time is None:
        start_time = '13:00'
    t_start_time = start_time
    start_time = parser.parse(start_time)

    if trip_dur is None:
        # get average in seconds
        trip_dur = _load_average_trip_duration()
    else:
        # convert hours into seconds
        trip_dur = trip_dur
    trip_dur = datetime.timedelta(hours=trip_dur)
    t_trip_dur = trip_dur.seconds

    if start_pos is None:
        start_pos = _init_start_position(graph)

    if end_pos is None:
        end_pos = _init_end_position(graph)

    # select start and end node base on start time and trip duration
    start_mpoi, end_mpoi = select_start_end_node(graph, start_time, trip_dur, start_pos, end_pos)

    # print 'start node: %d\tend node: %d' %(start_mpoi, end_mpoi) 

    return start_mpoi, end_mpoi, start_time, trip_dur


def _get_mpoi_qualities():
    spath = dump_base + '/micro_poi/mpoi_attractiveness.list'
    x = np.loadtxt(spath, skiprows=1)
    attract = x[:,0]
    lpop = x[:,1]
    spop = x[:,2]

    spath = dump_base + '/mpoi_network/mpoi_time.info'
    mpoi_time = np.loadtxt(spath)

    return attract, spop, lpop, mpoi_time


def _get_edges_weight(trip_path, paths):
    num_edges = len(trip_path)
    edges_weight = np.zeros(num_edges)
    # iterate over each of the edges and populate weight
    for i in range(num_edges):
        if i < num_edges-1:
            edges_weight[i] = paths[trip_path[i], trip_path[i+1]]
        else:
            edges_weight[i] = paths[trip_path[i], trip_path[0]]

    return edges_weight

def _get_edges_position(nodes_pos, path):
    num_edges = len(path)
    edges_pos = np.zeros((num_edges, 3, 2))
    # iterate over each of the edges and populate position
    for i, node in enumerate(path):
        edges_pos[i,:,0] = nodes_pos[i]
        if i < num_edges-1:
            edges_pos[i,:,1] = nodes_pos[i+1]
        else:
            edges_pos[i,:,1] = nodes_pos[0]


    return edges_pos


def find_ignore_ids(graph, start_time, trip_dur):
    ids = []

    # keep a 1 hour window before and after tour ends
    stime = start_time.hour + start_time.minute/60. - 1
    etime = stime + trip_dur.seconds/3600. + 2
    # iterate over all the nodes and remove nodes if there time is out of range
    for node in graph.vs:
        t = node['position'][2]
        if t < stime or t > etime:
            ids.append(node.index)

    return np.array(ids)


def _load_topic_distrib():
    spath = dump_base + '/mpoi_profiling/topic_distrib.h5'

    h5f = h5py.File(spath, 'r')
    topic_distrib = h5f['dataset_1'][:]
    h5f.close()

    return topic_distrib


def _load_topic_model():
    spath = dump_base + '/mpoi_profiling/lda_model/'

    fname = spath + 'model.pkl'

    model = joblib.load(fname)

    return model


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


def get_user_photos(uid, data):
    # get the user list
    users_list = data.iloc[:,8].values
    # get the photos list
    photos_list = data.iloc[:,6].values

    uids = users_list == uid

    uphotos = photos_list[uids]

    return uphotos

def get_user_features(u_photos):

    features = np.zeros(n_clusters)
    for pid in u_photos:
        doc = col_seg_fv.find_one({'_id': pid})
        assert doc
        ids = np.fromiter(doc['seg_ids'], dtype=int)
        freq_count = np.bincount(ids, minlength=n_clusters)
        features += freq_count

    return features

def compute_user_interest(uid, model):
    # load data from disk
    data = get_data()

    u_photos = get_user_photos(uid, data)

    user_features = get_user_features(u_photos)

    # find topic distribution
    topic_dist = model.transform(user_features.reshape(1,-1))

    return topic_dist

def _compute_liking(uid):
    # get mpoi topic distribtuion
    topic_distrib = _load_topic_distrib()
    model = _load_topic_model()

    # get user topic distribution
    user_tdistrib = compute_user_interest(uid, model)

    # compute cosine similarity
    liking = cosine_similarity(user_tdistrib, topic_distrib)[0]

    return liking/max(liking)

def find_profitability(gains, stay_times, graph, start_time, trip_dur, method_use, uid=None):

    # find ignore node ids based on start time and trip duration
    ignore_ids = find_ignore_ids(graph, start_time, trip_dur)

    # consider only those micor-poi which have stay time longer than 60 seconds
    pi_mpois = np.array([g/s if s > 60 and g < 500 else 0 for g,s in zip(gains, stay_times)])
    pi_mpois[ignore_ids] = 0.
    pi_mpois = pi_mpois/max(pi_mpois)

    # reduce the profitability of out of range micropois
    if method_use == method.proposed or method_use == method.personal or method_use == method.social:
        spop[ignore_ids] = 0
        lpop[ignore_ids] = 0

    # find total profitability
    pi = None
    sorted_ids = None

    if method_use == method.proposed:
        pi = delta*pi_mpois + kappa*spop + theta*lpop
        sorted_ids = np.argsort(pi)[::-1]
    elif method_use == method.random:
        sorted_ids = np.random.permutation(gains.size)
    elif method_use == method.social:
        pi = kappa*spop
        sorted_ids = np.argsort(pi)[::-1]
    elif method_use == method.local:
        pi = theta*lpop
        sorted_ids = np.argsort(pi)[::-1]
    elif method_use == method.socloc:
        pi = kappa*spop + theta*lpop
        sorted_ids = np.argsort(pi)[::-1]
    elif method_use == method.profit:
        pi = delta*pi_mpois
        sorted_ids = np.argsort(pi)[::-1]
    elif method_use == method.psoc:
        pi = delta*pi_mpois + kappa*spop
        sorted_ids = np.argsort(pi)[::-1]
    elif method_use == method.ploc:
        pi = delta*pi_mpois + theta*lpop
        sorted_ids = np.argsort(pi)[::-1]
    elif method_use == method.personal:
        mpoi_liking = _compute_liking(uid)
        pi = delta*pi_mpois + kappa*spop + theta*lpop + eta*mpoi_liking
        sorted_ids = np.argsort(pi)[::-1]
    
    return pi_mpois, sorted_ids

def _plot(path, v_pos, e_pos, sname, attract):
    plt.plot(v_pos[:,1], v_pos[:,0])
    area = np.pi*(100)
    colors = attract
    plt.scatter(v_pos[:,1], v_pos[:,0], s=area, c=colors, alpha=0.5)
    plt.savefig(sname)
    plt.close('all')


def _dump_path(trip_path, graph, stay_list):
    mpoi_pos = []

    for i, node in enumerate(trip_path):
        pos_3d = graph.vs[node]['position']
        assert node == graph.vs[node].index
        mpoi_pos.append(pos_3d[:2])

    mpoi_pos = np.array(mpoi_pos)
    data = np.hstack([mpoi_pos, stay_list.reshape(-1,1)])

    results_base = dump_base_nas + '/results/path_data/'
    if not os.path.exists(results_base):
        os.makedirs(results_base)

    # s_name = results_base + _poi + '_' + str(t_start_time) + '_' + str(t_trip_dur) + '_' + str(iter) + '_tour_path.html'
    s_name = results_base + _poi + '_' + str(t_start_time.split(':')[0]) + '_' + str(t_trip_dur) + '_tour_path.list'
    print s_name

    np.savetxt(s_name, data, fmt='%.8f', delimiter='\t')

def plot_path(trip_path, graph, paths, start_mpoi, path_list):
    n_nodes = len(trip_path)
    mpoi_pos = np.zeros(shape=(n_nodes,3))
    attr = np.zeros(n_nodes)
    pop = np.zeros(n_nodes)

    mpos = []
    node_list = []

    for i, node in enumerate(trip_path):
        pos_3d = graph.vs[node]['position']
        assert node == graph.vs[node].index
        mpoi_pos[i,:] = pos_3d[:3]
        attr[i] = attract[node]
        pop[i] = lpop[node]
        # if i < n_nodes-1:
        #     print node, trip_path[i+1]
        #     print path_list[node][trip_path[i+1]]
        #     for j in path_list[node][trip_path[i+1]]:
        #         if j not in node_list:
        #             node_list.append(j)

    edges_pos = _get_edges_position(mpoi_pos, trip_path)
    edges_weight = _get_edges_weight(trip_path, paths)

    # trip_path = update_trip_path(node_list, paths, graph)
    # s_id = np.where(trip_path == start_mpoi)[0][0]
    # trip_path = rotate_list(trip_path, -s_id)

    # for i, node in enumerate(trip_path):
    #     pos_3d = graph.vs[node]['position']
    #     assert node == graph.vs[node].index
    #     mpos.append(pos_3d)

    results_base = dump_base_nas + '/results/path/'
    if not os.path.exists(results_base):
        os.makedirs(results_base)

    # s_name = results_base + _poi + '_' + str(t_start_time) + '_' + str(t_trip_dur) + '_' + str(iter) + '_tour_path.html'
    s_name = results_base + _poi + '_' + str(t_start_time.split(':')[0]) + '_' + str(t_trip_dur) + '_tour_path.png'
    print s_name

    # plot_network.plot(trip_path, mpoi_pos, edges_pos, edges_weight, attract, pop, mpoi_time, _poi, s_name)
    # _plot(trip_path, mpoi_pos, edges_pos, s_name, attr)
    _plot(trip_path, mpoi_pos, np.array(mpos), s_name, attr)


def update_trip_path(trip_mpois, paths, graph):
    """
        solve TSP problem for finding the shortest path through all 
        the nodes in this trip 
    """
    n_nodes = len(trip_mpois)
    # adjacency matrix
    new_paths = np.zeros(shape=(n_nodes, n_nodes))

    # iterate through all the nodes and create a list of nodes with sequential id
    for i, node1 in enumerate(trip_mpois):
        for j, node2 in enumerate(trip_mpois):
            new_paths[i, j] = paths[node1, node2]

    # new_paths = new_paths/np.max(new_paths[new_paths < _INF])
    # new_paths[np.isinf(new_paths)] = _INF

    # create a dummy edge between end and start node with weight 0
    new_paths[1,0] = -_INF
    # new_paths[0,1] = _INF

    shortest_path = None
    if n_nodes > 5:
        shortest_path, dist = tsp.solve(n_nodes, new_paths)
        # shortest_path = range(n_nodes)
    else:
        shortest_path = range(n_nodes)

    trip_path = np.array(trip_mpois)[shortest_path]

    if ___DEBUG:
        fname = 'dump/' + str(n_nodes) + '.dist'
        np.savetxt(fname, new_paths, fmt='%.6f')
    
        mpoi_pos = np.zeros(shape=(n_nodes,2))
    
        for i, node in enumerate(trip_mpois):
            pos_3d = graph.vs[node]['position']
            assert node == graph.vs[node].index
            mpoi_pos[i,:] = pos_3d[:2]

        fname = 'dump/' + str(n_nodes) + '.pos'
        np.savetxt(fname, mpoi_pos)
    
    # print trip_mpois, trip_path

    return trip_path


def _slope(params, x, r):
    [a, b, c] = params
    m = (a*np.log(x)+c)/(r+x)

    return m


def _load_mean_params():

    spath = dump_base + '/micro_poi/mean_params.list'
    model_params = np.loadtxt(spath)

    return model_params

def find_stay_time(params, rtime, stime, mpoi_gain, stay_offset):
    [a, b, c] = params
    r = rtime
    s0 = stime
    gain = 0.
    # assume that we need to increase the stay time for optimal stay time
    incr = 1

    # only compute for positive slopes
    if a < 0 or a*np.log(4000)+c > 1000:
        params = _load_mean_params()
        [a, b, c] = params

    if rtime > 0:
        if s0 < np.exp(-c/a):
            s0 += np.exp(-c/a)
        m0 = _slope(params, s0, r)
        m1 = _slope(params, s0+1, r)

        # find out which way to move
        if m1 > m0:
            incr = 1
        else:
            incr = -1

        # now move to the tangent point in the curve
        s0 += incr
        m1 = _slope(params, s0, r)
        while m1 > m0:
            s0 += incr
            m0 = m1
            m1 = _slope(params, s0, r)
    else:
        # for start node
        s0 = stime
    
    s0 += stay_offset*s0/100.
    if s0 > np.exp(-c/a):
        gain = a*np.log(s0)+c
    else:
        if stay_offset == 0:
            gain = mpoi_gain
        else:
            gain = 0.

    return s0, gain

# def update_trip_time(trip_path, gains, model_params):
def update_trip_time(trip_path, paths, stay_time, mpoi_gains, start_end, model_params, method_use, stay_offset):
    """
        update the trip time based on the current path
    """

    trip_time = 0.0
    tot_gain = 0.
    time_list = []
    stay_list = []
    gain_list = []

    for idx, node in enumerate(trip_path):
        next_node = trip_path[(idx+1)%trip_path.size]
        rtime = paths[node, next_node]
        trip_time += rtime
        time_list.append(rtime)

        # if this is start node or end node check if it is in the tour
        if next_node in start_end and not start_end[next_node]:
            # don't add stay time
            gain_list.append(0)
            stay_list.append(0)
        else:
            # compute stay time
            if method_use == method.proposed or method_use == method.personal or method_use == method.profit:
                stime, gain = find_stay_time(model_params[next_node], rtime, stay_time[next_node], mpoi_gains[next_node], stay_offset)
            else:
                stime = stay_time[next_node]
                gain = mpoi_gains[next_node]
            trip_time += stime
            tot_gain += gain

            stay_list.append(stime)
            gain_list.append(gain)
        
    return trip_time, tot_gain, time_list, stay_list, gain_list

def _modify_paths(paths):

    p = np.zeros(shape=paths.shape)
    n_nodes = paths.shape[0]
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if i == j:
                p[i,j] = 0
            else:
                p[i,j] = min(paths[i,j], paths[j,i])
                p[j,i] = p[i,j]

    return p

def _compute_min_dist(mpid, mpoi_list, graph):

    min_dist = _INF
    for mpoi in mpoi_list:
        p1 = graph.vs[mpid]['position']
        p2 = graph.vs[mpoi]['position']

        dist = _distance(p1[:2], p2[:2])
        if dist < min_dist:
            min_dist = dist


    return min_dist


def invoke_tour_prediction(data, graph, start_time, trip_dur, start_mpoi, end_mpoi, method_use, stay_offset=0, uid=None):
    paths = data[0]
    path_list = data[1]
    mpoi_gains = data[2]
    stay_time = data[3]
    reach_time = data[4]
    model_params = data[5]

    # find the pi value for each mpoi (rate of gain)
    pi_mpois, sorted_ids = find_profitability(mpoi_gains, stay_time, graph, start_time, trip_dur, method_use, uid=uid)

    # remove inf values from path
    paths[np.isinf(paths)] = np.max(paths[np.isfinite(paths)])

    # make the graph undirected
    # paths = _modify_paths(paths)
    paths[end_mpoi, start_mpoi] = 0.

    # use the actual distance between nodes
    geo_paths = update_node_dist(graph)

    trip_time = 0.0
    trip_mpois = None
    trip_path = None
    start_end = None
    time_list = None
    stay_list = None
    gain_list = None
    gain = 0
    if start_mpoi != end_mpoi:
        trip_mpois = [start_mpoi, end_mpoi]
        trip_path = [start_mpoi, end_mpoi]
        # don't count start and end node while computing travel time
        # unless they are included in the path
        start_end = {start_mpoi:False, end_mpoi:False}
    else:
        trip_mpois = [start_mpoi]
        trip_path = [start_mpoi]
        # don't count start and end node while computing travel time
        # unless they are included in the path
        start_end = {start_mpoi:False}

    # iterate over the sorted mpois based on profitability untill time exhausted
    for i, mpoi_id in enumerate(sorted_ids):
        # if this mpoi is close enough to any other mpoi ignore this
        t0 = time.time()
        min_dist = _compute_min_dist(mpoi_id, trip_mpois, graph)
        if min_dist < _MIN_DIST:
            continue

        if mpoi_id == start_mpoi:
            # mark start node to be included in the path
            start_end[start_mpoi] = True
            trip_time += stay_time[mpoi_id]
        elif mpoi_id == end_mpoi:
            # mark the end node to be included in the path
            start_end[end_mpoi] = True
            trip_time += stay_time[mpoi_id]
        else:
            trip_mpois.append(mpoi_id)
            trip_path = update_trip_path(trip_mpois, geo_paths, graph)
            trip_time, gain, time_list, stay_list, gain_list = update_trip_time(trip_path, paths, stay_time, mpoi_gains, start_end, model_params, method_use, stay_offset)

        print i, time.time() - t0

        if __DEBUG:
            print trip_time
            print trip_path

        if trip_time > trip_dur.seconds and len(trip_path) > 5:
            break

    # first rotate the path such that the first poi is start
    s_id = np.where(trip_path == start_mpoi)[0][0]

    trip_path = rotate_list(trip_path, -s_id)
    time_list = rotate_list(time_list, -s_id)
    stay_list = rotate_list(stay_list, -s_id)
    gain_list = rotate_list(gain_list, -s_id)

    if _DEBUG and t_trip_dur:
        pass
        # plot_path(trip_path, graph, geo_paths, start_mpoi, path_list)
        # _dump_path(trip_path, graph, np.array(stay_list))

    if _DEBUG:
        print trip_path, trip_time, gain

    return trip_path, trip_time, gain, time_list, stay_list, gain_list


def rotate_list(l, i):
    path = deque(l)
    path.rotate(i)
    trip_path = list(path)

    return trip_path

def predict(poi, start_time, trip_dur, start_mpoi, end_mpoi, method_use, stay_offset=0, uid=None):

    if __DEBUG:
        print 'initializing...'
    _init(poi)

    # load data
    data = load_data()

    # load the network
    graph = _load_network()

    trip_path, trip_time, gain, time_list, stay_list, gain_list = invoke_tour_prediction(data, graph, start_time, trip_dur, start_mpoi, end_mpoi, method_use, stay_offset=stay_offset, uid=uid)

    if __DEBUG:
        print trip_path

    ntrip = Trip()
    ntrip.mpoi_path = trip_path
    ntrip.trip_time = datetime.timedelta(seconds=trip_time)
    ntrip.gain = gain
    ntrip.time = time_list
    ntrip.stay_time = stay_list
    ntrip.gain_list = gain_list
    
    # print trip_path, trip_time, gain

    return ntrip


def master(poi, start_time=None, trip_dur=None, start_pos=None, end_pos=None, method_use=method.proposed):

    if __DEBUG:
        print 'initializing...'
    _init(poi)

    # load data
    data = load_data()

    # load the network
    graph = _load_network()

    for trip_dur in range(1,7):
        # invoke diet algorithm
        # initialize 
        t0 = time.time()
        start_time = "13:00"
        start_mpoi, end_mpoi, start_time, trip_dur = _init_opt(graph, start_time, trip_dur, start_pos, end_pos)
        start_mpoi = 94
        end_mpoi = 84

        invoke_tour_prediction(data, graph, start_time, trip_dur, start_mpoi, end_mpoi, method_use)

        print 'time required: ', time.time() - t0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_name"

    poi = str(sys.argv[1])

    if ___DEBUG:
        print 'starting master...'
    master(poi)

