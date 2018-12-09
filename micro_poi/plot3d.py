import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot(X):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = 100

    xs, ys, zs = X[:,0], X[:,1], X[:,2]
    ax.scatter(xs, ys, zs)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    plt.close()

