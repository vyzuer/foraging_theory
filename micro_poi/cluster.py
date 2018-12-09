import sys
import dbscan
import ap
import gmm
import numpy as np


def cluster_dbscan(db_path, file_name):
    dbscan.cluster(db_path, file_name)

def cluster_ap(db_path, file_name):
    ap.cluster(db_path, file_name)


