import sys
import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

def cluster(num_samples, num_clusters):
    x = 3.3*np.random.randn(num_samples,2)
    X = StandardScaler().fit_transform(x)

    flag = False
    
    if flag:
        t0 = time.time()
        km = MiniBatchKMeans(init='k-means++', n_clusters=num_clusters, batch_size=3*num_clusters,
                max_no_improvement=10, verbose=0, max_iter=100,
                random_state=0)
        km.fit(X)
        t1 = time.time()

        print 'kmeans time taken : ', t1 - t0

    flag = True
    if flag:
        t0 = time.time()
        bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=10000)
        # bandwidth = 0.5
        print bandwidth
        print 'bandwidth estimation time : ', time.time() - t0

        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, min_bin_freq=100, n_jobs=-1)
        ms.fit(X)
        t1 = time.time()
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_)
        print 'meanshift time taken : ', t1 - t0

    flag = False
    if flag:
        x = 3.3*np.random.randn(num_samples,2)
        X = StandardScaler().fit_transform(x)
    
        t0 = time.time()
        db = DBSCAN(eps=0.3, min_samples=100)
        db.fit(X)
        t1 = time.time()
        n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        print 'number of clusters:', n_clusters_

        print 'DBSCAN time taken : ', t1 - t0

    flag = False

    if flag:
        t0 = time.time()
        bm = Birch(threshold=0.3, n_clusters=None)
        bm.fit(X)
        t1 = time.time()
        print 'number of clusters:', np.unique(bm.labels_).size

        print 'Birch time taken : ', t1 - t0

if __name__ == '__main__':
    num_samples = int(sys.argv[1])
    num_clusters = int(sys.argv[2])

    cluster(num_samples, num_clusters)

