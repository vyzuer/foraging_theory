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
base_dir = None

# add the code package to the path
sys.path.append('/home/vyzuer/work/code/foraging_theory/')

import common.globals as gv

def plot(trip_path, nodes_pos, edges_pos, edges_w, attract, pop, mpoi_time, poi, fname):

    labels = [str(i) for i in trip_path.tolist()]
    weights = [str(i) for i in edges_w.tolist()]
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
                       line=Line(color='rgb(125,125,125)', width=1),
                       text=weights,
                       hoverinfo='text'
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
                       hoverinfo='text',
                       opacity=0.8
                       )
    
    axis = dict(showbackground=True,
                showline=True,
                zeroline=True,
                showgrid=True,
                showticklabels=True,
                title=''
                )
    
    layout = Layout(title="Predicted Tour Path",
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
                    hovermode='closest')
    
    data = Data([trace1, trace2])
    fig = Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=fname, auto_open=False)


