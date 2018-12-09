from pymongo import MongoClient
import numpy as np
import sys
import pandas as pd
import os
from pprint import pprint
import datetime
from dateutil import parser
import matplotlib.pyplot as plt

_DEBUG_MODE = True

sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv

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


def time_analysis(dump_base, fp, data):
    cap_date = data.iloc[:,1].values
    struct_date = [parser.parse(d) for d in cap_date]

    year_list = [d.year for d in struct_date]
    month_list = [d.month for d in struct_date]
    hour_list = [d.hour for d in struct_date]
    day_list = [d.weekday() for d in struct_date]

    # print year_list[1], month_list[1], hour_list[1], day_list[0]

    fp.write('\n----------- time analysis-----------\n')
    fp.write('%10s\t%10s\t%10s\n' %('', 'year', 'hour'))
    fp.write('%10s\t%10d\t%10d\n' %('min', np.min(year_list), np.min(hour_list)))
    fp.write('%10s\t%10d\t%10d\n' %('max', np.max(year_list), np.max(hour_list)))

    
    plt.hist(year_list)
    plt.xlabel('year')
    plt.ylabel('number of photos')

    # save the plot
    year_hist = dump_base + 'year_hist.png'
    plt.savefig(year_hist)
    plt.clf()

    plt.hist(hour_list, bins=24)
    plt.xlabel('hour')
    plt.ylabel('number of photos')

    # save the plot
    hour_hist = dump_base + 'hour_hist.png'
    plt.savefig(hour_hist)
    plt.clf()

    plt.hist(month_list, bins=12)
    plt.xlabel('month')
    plt.ylabel('number of photos')

    # save the plot
    month_hist = dump_base + 'month_hist.png'
    plt.savefig(month_hist)
    plt.clf()

    plt.hist(day_list, bins=7)
    plt.xlabel('day')
    plt.ylabel('number of photos')

    # save the plot
    day_hist = dump_base + 'day_hist.png'
    plt.savefig(day_hist)
    plt.clf()


def sm_analysis(dump_base, fp, data):
    # get the views list
    view_list = data.iloc[:,9].values.astype(int)
    favs_list = data.iloc[:,2].values.astype(int)
    comm_list = data.iloc[:,0].values.astype(int)
    
    fp.write('\n----------- social media analysis---------\n')
    fp.write('%10s\t%10s\t%10s\t%10s\n' %('', 'views', 'favs', 'comments'))

    # total number of view/favs/comments
    t_view = np.sum(view_list)
    t_favs = np.sum(favs_list)
    t_comm = np.sum(comm_list)

    max_view = np.max(view_list)
    max_favs = np.max(favs_list)
    max_comm = np.max(comm_list)

    min_view = np.min(view_list)
    min_favs = np.min(favs_list)
    min_comm = np.min(comm_list)

    mean_view = np.mean(view_list)
    mean_favs = np.mean(favs_list)
    mean_comm = np.mean(comm_list)

    fp.write('%10s\t%10d\t%10d\t%10d\n' %('total', t_view, t_favs, t_comm))
    fp.write('%10s\t%10d\t%10d\t%10d\n' %('max', max_view, max_favs, max_comm))
    fp.write('%10s\t%10d\t%10d\t%10d\n' %('min', min_view, min_favs, min_comm))
    fp.write('%10s\t%10f\t%10f\t%10f\n' %('mean', mean_view, mean_favs, mean_comm))

    log_views = np.log2(view_list+1)+1
    bins = int(np.ceil(np.max(log_views)))

    plt.hist(log_views, bins=bins)
    plt.xlabel('log2(number_views+1)+1')
    plt.ylabel('number of photos')

    # save the plot
    views_hist = dump_base + 'views_hist.png'
    plt.savefig(views_hist)
    plt.clf()

    log_favs = np.log10(favs_list+1)+1
    bins = int(np.ceil(np.max(log_favs)))

    plt.hist(log_favs, 50)
    plt.xlabel('number of favorites')
    plt.ylabel('number of photos')

    # save the plot
    favs_hist = dump_base + 'favs_hist.png'
    plt.savefig(favs_hist)
    plt.clf()

    plt.hist(comm_list, 50)
    plt.xlabel('number of comments')
    plt.ylabel('number of photos')

    # save the plot
    comm_hist = dump_base + 'comm_hist.png'
    plt.savefig(comm_hist)
    plt.clf()


def user_analysis(dump_base, fp, data):
    # get the user list
    users_list = data.iloc[:,8].values
    
    dprint('sample user_id: %s' %( users_list[0]))
    
    # find unique users
    u_users, u_pos = np.unique(users_list, return_inverse=True)
    fp.write('\n---------- user information-----------\n')
    fp.write('Total number of unique users: %d\n' %(u_users.size))

    dprint('Total unique users: %d' %(u_users.size))

    # find the number of photos per user
    pp_user = np.bincount(u_pos)
    # dprint(('Photos per user: ', pp_user))

    # find maximum and minimum per user photos
    min_user = np.min(pp_user)
    max_user = np.max(pp_user)
    mean_user = np.mean(pp_user)
    fp.write('Minimum number of photos per user: %d\n' %(min_user))
    fp.write('Maximum number of photos per user: %d\n' %(max_user))
    fp.write('Mean number of photos per user: %d\n' %(mean_user))

    log_freq = np.log2(pp_user) + 1
    bins = int(np.ceil(np.max(log_freq)))
    dprint('number of bins: %d' %(bins))
    plt.hist(log_freq, bins=bins)
    plt.xlabel('log2(number_photos)+1')
    plt.ylabel('number of users')

    # save the plot
    user_info = dump_base + 'user_stat.png'
    plt.savefig(user_info)
    plt.clf()


def master(data, poi, clean=False):
    print 'performing data analysis...'
    # the dump path
    dump_base = gv.__base_dir + poi + '/data_analysis/'
    if not os.path.exists(dump_base):
        os.makedirs(dump_base)

    d_analysis = dump_base + 'data.analysis'
    if not clean and os.path.exists(d_analysis):
        print 'Data Analysis: database up to date. nothing to do.'
        return

    print 'Performing Data Anaysis...'
    fp = open(d_analysis, 'w')

    dprint(('sample row', data.iloc[:1,:]))

    n_photos = data.shape[0]
    fp.write('Total number of photos: %d\n' %(n_photos))
    dprint('total number of photos: %d' %(n_photos))

    # user analysis
    user_analysis(dump_base, fp, data)

    # social media analysis
    sm_analysis(dump_base, fp, data)

    # time statistics
    time_analysis(dump_base, fp, data)

    fp.close()

    print 'data analysis done.'

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python data_analysis.py location_name"

    poi = str(sys.argv[1])

    # load the data
    data = get_data(poi)

    master(data, poi)

