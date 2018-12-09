import numpy as np
import sys
import pandas as pd
import os

_DEBUG_MODE = False

sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv

alpha = 0.003
beta = 0.1
gamma = 0.1

def dprint(mesg):
    if _DEBUG_MODE:
        print mesg

def get_data(poi):

    # the database for poi
    dataset_path = gv.__dataset_path + poi + '/poi_info.pkl'

    data = None

    try:
        data = pd.read_pickle(dataset_path)
    except IOError:
        print 'database error'
        exit(0)

    return data


def compute_pop_loc(labels, num_components):

    # count number of photos per mpoi
    num_poi = np.bincount(labels, minlength=num_components)
    dprint(num_poi)

    pop_loc = num_poi/float(max(num_poi))

    dprint('location popularity scores')
    dprint(pop_loc)

    return pop_loc

def _dump_a_scores(scores, poi):

    dump_base = gv.__base_dir + poi + '/micro_poi/'
    if not os.path.exists(dump_base):
        print 'Error: database error. micro_poi directory'
        exit(0)

    f_scores = dump_base + 'scores.list'
    np.savetxt(f_scores, scores, fmt='%f')


def compute_pop_sm(labels, num_components, data, poi):
    # pop_asm : average social media popularity
    # pop_csm : cumulative social media popularity

    # get the views/favs/comm list
    vfc_list = data.iloc[:, [9,2,0]].astype(int).values

    # pop_asm
    pop_asm = np.zeros(num_components)
    a_scores = 1 - 1/np.exp(alpha*vfc_list[:,0] + beta*vfc_list[:,1], gamma*vfc_list[:,2])

    _dump_a_scores(a_scores, poi)

    for i in range(num_components):
        vfc = a_scores[labels == i]

        if vfc != []:
            pop_asm[i] = np.mean(vfc)

    dprint('average social media popularity scores')
    dprint(pop_asm)

    # first compute the aesthetic scores


    # pop_csm
    vfc_data = np.zeros(shape=(num_components, 3))

    for i in range(num_components):
        vfc = vfc_list[labels == i, :]

        vfc_data[i,:] = np.sum(vfc, axis=0)
        
    # normalize and average out the scores
    pop_csm = np.mean(vfc_data/np.max(vfc_data, axis=0), axis=1)

    dprint('cumulative social media popularity scores')
    dprint(pop_csm)

    return pop_asm, pop_csm
    

def master(labels, num_components, data, poi, clean=False):
    # the dump path
    dump_base = gv.__base_dir + poi + '/micro_poi/'
    if not os.path.exists(dump_base):
        print 'Error: database error. micro_poi directory'
        exit(0)

    mpoi_scores = dump_base + 'mpoi_attractiveness.list'
    if not clean and os.path.exists(mpoi_scores):
        print 'MPOI Evaluation: database up to date. nothing to do.'
        return

    print 'MPOI Eval: performing MPOI evaluation...'
    fp = open(mpoi_scores, 'w')

    dprint(('sample row', data.iloc[:1,:]))

    # compute localtion popularity
    pop_loc = compute_pop_loc(labels, num_components)

    # compute the average and cumulative social media scores
    pop_asm, pop_csm = compute_pop_sm(labels, num_components, data, poi)

    # dump the scores
    attractiveness = np.mean([pop_loc, pop_asm, pop_csm], axis=0)
    dprint('attractiveness scores')
    dprint(attractiveness)

    scores = np.column_stack((attractiveness, pop_loc, pop_asm, pop_csm))

    # header of the file
    fp.write('%10s %10s %10s %10s\n' %('attractiveness', 'spatial_popularity', 'average_sm_pop', 'cumulative_sm_pop'))

    np.savetxt(fp, scores, fmt='%f')

    fp.close()

    print 'mpoi evaluation done.'

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python mpoi_evaluation.py location_name"

    poi = str(sys.argv[1])

    # load the data
    data = get_data(poi)

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

    master(labels, num_components, data, poi, clean=True)

