from pymongo import MongoClient
import numpy as np
import sys
import pandas as pd
import os
from pprint import pprint
import datetime
from dateutil import parser
import gmm
import plot3d


""" ouput dump
micro_poi/gmm/bic.scores
micro_poi/gmm/bic.png
micro_poi/gmm/model/*
micro_poi/gmm/scaler/*
micro_poi/labels.list
micro_poi/mpoi_attractiveness.list
data_analysis/*

"""
client = None
db_foraging = None

# add the code package to the path
sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv
import data_analysis as da
import mpoi_evaluation as mpoi_eval

def load_globals():
    global client
    global db_foraging

    client = MongoClient()

    db_foraging = client.ysr_foraging_db

def get_geo_coords(poi):
    col = db_foraging.pois_geo

    doc = col.find_one({"_id":poi})

    lat0 = doc['lat0']
    lat1 = doc['lat1']
    lon0 = doc['lon0']
    lon1 = doc['lon1']

    return lat0, lon0, lat1, lon1


def create_dataset(dataset_path, poi):
    # load the dataset
    poi_col = db_foraging[poi]

    items = poi_col.find()
    
    # new dataframe to store data
    data = []

    # iterate over the database to create a new set
    for doc in items:
        # pprint(doc)

        # convert the time to hours format to use as a feature
        dt = doc['Date_taken']
        dt = parser.parse(dt)
        cap_time = dt.hour + dt.minute/60.0

        n_views, n_favs, n_comments = 0,0,0
        # all the photos do not have photo_info and favorites_info so check
        try:
            n_views = doc['photo_info']['views']
            n_comments = doc['photo_info']['comments']['_content']
            n_favs = doc['favorites_info']['total']
        except KeyError:
            pass

        data.append({'id':doc['_id'], 'photo_id':doc['Photo_id'], 'user_id':doc['User_NSID'], 'lat':doc['Latitude'], 'lon':doc['Longitude'], 'time':cap_time, 'views':n_views, 'favs':n_favs, 'comments':n_comments, 'date':doc['Date_taken']})

    df = pd.DataFrame(data)

    # dump the data
    df.to_pickle(dataset_path)

    return df

def get_data(poi):

    # the database for poi
    dataset_path = gv.__dataset_path + poi + '/poi_info.pkl'

    data = None
    # first check if data is present otherwise create one
    if not os.path.exists(dataset_path):
        data = create_dataset(dataset_path, poi)
    else:
        data = pd.read_pickle(dataset_path)

    return data

def identify_mpoi(data, poi, clean=False):
    # collect the latitude, longitude and time information for clustering
    llt_info = data.iloc[:,[4,5,7]].values
    print llt_info.shape

    # plot for testing
    # plot3d.plot(llt_info)

    labels = None
    num_components = None

    # if labels file is not present perform gmm
    poi_base_dir = gv.__base_dir + poi + '/micro_poi/'
    if not os.path.exists(poi_base_dir):
        os.makedirs(poi_base_dir)

    labels_file = poi_base_dir + 'labels.list'

    if clean or not os.path.exists(labels_file):
        print 'MPOI identification: performing gmm analysis...'
        # perform clustering/gmm
        labels, num_components = gmm.gmm_1(llt_info, poi_base_dir)
        np.savetxt(labels_file, labels, fmt='%d')
    else:
        print 'MPOI identification: database up to date. loading from disk'
        labels = np.loadtxt(labels_file, dtype='int')
        num_components = np.max(labels) + 1

    return labels, num_components

def master(poi, clean=False):
    # load the data
    data = get_data(poi)

    # perform clustering to identify micro pois
    labels, num_components = identify_mpoi(data, poi, clean)

    # perform data analysis for each of the locations
    da.master(data, poi, clean)

    # perform mpoi evaluation for quality
    mpoi_eval.master(labels, num_components, data, poi, clean)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python identify_mpoi.py location_name"

    poi = str(sys.argv[1])

    load_globals()

    master(poi, clean=True)

