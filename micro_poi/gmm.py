import itertools
import sys, os
import numpy as np
from scipy import linalg
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def gmm_1(X, poi_base_dir):
    
    # data preprocesing
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    gmm_dump_path = poi_base_dir + 'gmm/'
    bic_scores = gmm_dump_path + 'bic.scores'
    bic = np.loadtxt(bic_scores)

    n_components = np.argmin(bic) + 1

    best_gmm = None
    n_comp = n_components

    cv_type = 'spherical'

    best_gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
    best_gmm.fit(X)

    print 'covariance:%s, num_components:%d, bic: %f' %(cv_type, n_components, bic[n_components-1])

    # dump the bic score and plot
    gmm_dump_path = poi_base_dir + 'gmm/'

    # dump the gmm model
    model_dump = gmm_dump_path + "/model/"
    if not os.path.exists(model_dump):
        os.makedirs(model_dump)

    model_path = model_dump + "/gmm.pkl"

    scaler_dump = gmm_dump_path + "/scaler/"
    if not os.path.exists(scaler_dump):
        os.makedirs(scaler_dump)

    scaler_path = scaler_dump + "/scaler.pkl"

    joblib.dump(best_gmm, model_path)
    joblib.dump(scaler, scaler_path)

    # return the predicted labels
    labels = best_gmm.predict(X)

    return labels, n_comp


def gmm(X, poi_base_dir):
    
    # data preprocesing
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    lowest_bic = np.infty
    bic = []
    best_gmm = None
    n_comp = None
    n_components_range = range(1, 301)
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['full']

    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)

            bic.append(gmm.bic(X))
            print 'covariance:%s, num_components:%d, bic: %f' %(cv_type, n_components, bic[-1])

            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                n_comp = n_components
    
    bic = np.array(bic)
    
    # Plot the BIC scores
    plt.plot(n_components_range, bic)

    # dump the bi score and plot
    gmm_dump_path = poi_base_dir + 'gmm/'
    if not os.path.exists(gmm_dump_path):
        os.makedirs(gmm_dump_path)

    bic_plot = gmm_dump_path + 'bic.png'

    plt.savefig(bic_plot)

    bic_scores = gmm_dump_path + 'bic.scores'
    np.savetxt(bic_scores, bic, fmt='%d')
    
    # dump the gmm model
    model_dump = gmm_dump_path + "/model/"
    if not os.path.exists(model_dump):
        os.makedirs(model_dump)

    model_path = model_dump + "/gmm.pkl"

    scaler_dump = gmm_dump_path + "/scaler/"
    if not os.path.exists(scaler_dump):
        os.makedirs(scaler_dump)

    scaler_path = scaler_dump + "/scaler.pkl"

    joblib.dump(best_gmm, model_path)
    joblib.dump(scaler, scaler_path)

    # return the predicted labels
    labels = best_gmm.predict(X)

    return labels, n_comp

