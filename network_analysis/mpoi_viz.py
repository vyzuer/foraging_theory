import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import igraph as ig
import itertools
import sys
import os
import numpy as np
from sklearn import mixture
from sklearn.externals import joblib

__DEBUG = True
_DEBUG = True

dump_base = None
dump_base_nas = None
base_dir = None

# add the code package to the path
sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv


def load_globals(poi):
    global dump_base
    global dump_base_nas
    global base_dir
    global graph_flag

    base_dir = gv.__dataset_path + poi
    dump_base = gv.__base_dir + poi
    dump_base_nas = gv.__base_dir_nas + poi

def _init(poi):
    # load global variables
    load_globals(poi)

    np.set_printoptions(precision=4, suppress=True)


def _load_gmm():
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


def _load_graph():
    s_edges = dump_base + '/mpoi_network/mpoi_edges.info'

    assert os.path.exists(s_edges)

    edges = np.loadtxt(s_edges)
    num_nodes = edges.shape[0]
    nodes = np.arange(num_nodes)

    graph = ig.Graph.Weighted_Adjacency(edges.tolist())

    edges = graph.get_edgelist()

    return graph, nodes, edges


def _get_nodes_position(gmm, scaler, nodes):
    num_nodes = nodes.size

    nodes_pos = gmm.means_

    if _DEBUG:
        print 'number of nodes in network: ', num_nodes

    # this happened for wasmon, reason was
    # the last component was empty
    if nodes_pos.shape[0] != num_nodes:
        nodes_pos = nodes_pos[:num_nodes,:]

    nodes_pos = scaler.inverse_transform(nodes_pos)

    return nodes_pos


def _get_edges_position(nodes_pos, edges):
    num_edges = len(edges)
    edges_pos = np.zeros((num_edges, 3, 2))
    # iterate over each of the edges and populate position
    i = 0
    for e in edges:
        edges_pos[i,:,0] = nodes_pos[e[0]]
        edges_pos[i,:,1] = nodes_pos[e[1]]

        i+= 1

    return edges_pos


def _plot_network(nodes_pos, edges_pos, edges_w, attract, pop, mpoi_time, poi):

    width = np.log(10.0*edges_w).tolist()
    # labels = [str(i) for i in mpoi_time.tolist()]
    labels = [str(i) for i in range(mpoi_time.size)]
    size = 6 + np.log(500*pop)
    group = np.log(100*attract+1)

    Xe = []
    Ye = []
    Ze = []
    
    for e in edges_pos.tolist():
        Xe += [e[0][0],e[0][1], None]
        Ye += [e[1][0],e[1][1], None]
        Ze += [e[2][0],e[2][1], None]

    trace1 = Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       line=Line(color='rgb(125,125,125)', width=width),
                       hoverinfo='none'
                       )

    trace2 = Scatter3d(x=nodes_pos[:,0],
                       y=nodes_pos[:,1],
                       z=nodes_pos[:,2],
                       mode='markers',
                       name='actors',
                       marker=Marker(symbol='dot',
                           size=size,
                           color=group,
                           colorscale='Viridis',
                           line=Line(color='rgb(50,50,50)', width=0.5)
                           ),
                       text=labels,
                       hoverinfo='text'
                       )
    
    axis = dict(showbackground=True,
                showline=True,
                zeroline=True,
                showgrid=True,
                showticklabels=True,
                title=''
                )
    
    layout = Layout(title="Network of mpoi's",
                    width=1000,
                    height=1000,
                    showlegend=True,
                    scene=Scene(
                        xaxis=XAxis(axis),
                        yaxis=YAxis(axis),
                        zaxis=ZAxis(axis),
                        ),
                    margin=Margin(
                        t=100
                        ),
                    hovermode='closest',
                    annotations=Annotations([
                        Annotation(
                            showarrow=False,
                            text=poi,
                            xref='paper',
                            yref='paper',
                            x=0,
                            y=0.1,
                            xanchor='left',
                            yanchor='bottom',
                            font=Font(
                                size=14
                                )
                            )
                        ]),    )
    
    data = Data([trace1, trace2])
    fig = Figure(data=data, layout=layout)

    base = dump_base_nas + '/mpoi_network/'
    if not os.path.exists(base):
        os.makedirs(base)
    filename = base + poi + '_mpoi_network.html'
    plotly.offline.plot(fig, filename=filename, auto_open=False)


def _get_mpoi_qualities():
    spath = dump_base + '/micro_poi/mpoi_attractiveness.list'
    attract = np.loadtxt(spath, skiprows=1)[:,0]

    spath = dump_base + '/mpoi_network/photo_freq.info'
    pop = np.loadtxt(spath)

    spath = dump_base + '/mpoi_network/mpoi_time.info'
    mpoi_time = np.loadtxt(spath)

    return attract, pop, mpoi_time

def master(poi):
    # load the gmm and scaler model for mpoi
    if _DEBUG:
        print 'loading gmm model from dump...'
    gmm, scaler = _load_gmm()

    # load the network information
    if _DEBUG:
        print 'loading graph info from dump...'
    graph, nodes, edges = _load_graph()

    # get the nodes position from gmm model
    if _DEBUG:
        print 'getting nodes position...'
    nodes_pos = _get_nodes_position(gmm, scaler, nodes)

    # get the edges position using the edges information
    if _DEBUG:
        print 'getting edges position...'
    edges_pos = _get_edges_position(nodes_pos, edges)

    # load the mpoi qualities
    attrac, pop, mpoi_time = _get_mpoi_qualities()

    # plot the 3d network
    if _DEBUG:
        print 'plotting network...'
    edges_w = np.array(graph.es["weight"])
    _plot_network(nodes_pos, edges_pos, edges_w, attrac, pop, mpoi_time, poi)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: python sys.argv[0] location_name"

    poi = str(sys.argv[1])

    if _DEBUG:
        print 'initializing...'
    _init(poi)

    if _DEBUG:
        print 'starting master...'
    master(poi)


