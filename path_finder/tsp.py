#!/usr/bin/env python

""" Traveling salesman problem solved using Simulated Annealing.
"""
from scipy import *
from pylab import *
import numpy as np

_DEBUG = False

def Distance(R1, R2):
    return sqrt((R1[0]-R2[0])**2+(R1[1]-R2[1])**2)

def TotalDistance(city, R):
    dist=0
    for i in range(len(city)-1):
        dist += Distance(R[city[i]],R[city[i+1]])
    dist += Distance(R[city[-1]],R[city[0]])
    return dist
    
def Total_Distance(city, edge_matrix):
    dist=0
    for i in range(len(city)-1):
        dist += edge_matrix[city[i], city[i+1]]
    dist += edge_matrix[city[-1], city[0]]

    return dist

def reverse(city, n):
    nct = len(city)
    newcity = np.copy(city)
    nn = (1+ ((n[1]-n[0]) % nct))/2 # half the lenght of the segment to be reversed
    # the segment is reversed in the following way n[0]<->n[1], n[0]+1<->n[1]-1, n[0]+2<->n[1]-2,...
    # Start at the ends of the segment and swap pairs of cities, moving towards the center.
    for j in range(nn):
        k = (n[0]+j) % nct
        l = (n[1]-j) % nct
        (newcity[k],newcity[l]) = (city[l],city[k])  # swap

    return newcity
    
def transpt(city, n):
    nct = len(city)
    
    newcity=[]
    # Segment in the range n[0]...n[1]
    for j in range( (n[1]-n[0])%nct + 1):
        newcity.append(city[ (j+n[0])%nct ])
    # is followed by segment n[5]...n[2]
    for j in range( (n[2]-n[5])%nct + 1):
        newcity.append(city[ (j+n[5])%nct ])
    # is followed by segment n[3]...n[4]
    for j in range( (n[4]-n[3])%nct + 1):
        newcity.append(city[ (j+n[3])%nct ])
    return newcity

def Plot(city, R, dist):
    # Plot
    Pt = [R[city[i]] for i in range(len(city))]
    Pt += [R[city[0]]]
    Pt = array(Pt)
    title('Total distance='+str(dist))
    plot(Pt[:,0], Pt[:,1], '-o')
    show()


def solve(n_nodes, edge_matrix):
    """
        this function solves the tsp problem
        input: number of nodes and their edge matrix
        output: the shortest path
    """

    maxTsteps = 100    # Temperature is lowered not more than maxTsteps
    Tstart = 0.2      # Starting temperature - has to be high enough
    fCool = 0.9        # Factor to multiply temperature at each cooling step
    maxSteps = 100*n_nodes     # Number of steps at constant temperature
    maxAccepted = 10*n_nodes   # Number of accepted steps at constant temperature

    Preverse = 0.5      # How often to choose reverse/transpose trial move

    # The index table -- the order the cities are visited.
    city = range(n_nodes)

    # Distance of the travel at the beginning
    dist = Total_Distance(city, edge_matrix)

    best_path = city
    best_dist = dist

    # Stores points of a move
    n = zeros(6, dtype=int)
    nct = n_nodes # number of cities
    
    T = Tstart # temperature

    # Plot(city, R, dist)
    
    for t in range(maxTsteps):  # Over temperature

        accepted = 0
        for i in range(maxSteps): # At each temperature, many Monte Carlo steps
            
            while True: # Will find two random cities sufficiently close by
                # Two cities n[0] and n[1] are choosen at random
                n[0] = int((nct)*rand())     # select one city
                n[1] = int((nct-1)*rand())   # select another city, but not the same
                if (n[1] >= n[0]): n[1] += 1   #
                if (n[1] < n[0]): (n[0],n[1]) = (n[1],n[0]) # swap, because it must be: n[0]<n[1]
                nn = (n[0]+nct -n[1]-1) % nct  # number of cities not on the segment n[0]..n[1]
                if nn>=3: break
        
            # We want to have one index before and one after the two cities
            # The order hence is [n2,n0,n1,n3]
            n[2] = (n[0]-1) % nct  # index before n0  -- see figure in the lecture notes
            n[3] = (n[1]+1) % nct  # index after n2   -- see figure in the lecture notes
            
            if Preverse > rand(): 
                # Here we reverse a segment
                # What would be the cost to reverse the path between city[n[0]]-city[n[1]]?
                new_city = reverse(city, n)
                d1 = Total_Distance(new_city, edge_matrix)
                de = d1 - dist

                if de<0 or exp(-de/T)>rand(): # Metropolis
                    accepted += 1
                    # dist += de
                    city = new_city
                    dist = d1
                    if dist < best_dist:
                        best_path = city
                        best_dist = dist
            else:
                # Here we transpose a segment
                nc = (n[1]+1+ int(rand()*(nn-1)))%nct  # Another point outside n[0],n[1] segment. See picture in lecture nodes!
                n[4] = nc
                n[5] = (nc+1) % nct
        
                # Cost to transpose a segment
                new_city = transpt(city, n)
                d1 = Total_Distance(new_city, edge_matrix)
                de = d1 - dist
                
                if de<0 or exp(-de/T)>rand(): # Metropolis
                    accepted += 1
                    # dist += de
                    city = new_city
                    dist = d1
                    
                    if dist < best_dist:
                        best_path = city
                        best_dist = dist
            if accepted > maxAccepted: break

        if _DEBUG:
            print "T=%10.5f , distance= %10.5f , accepted steps= %d" %(T, dist, accepted)
        T *= fCool             # The system is cooled down
        if accepted == 0: break  # If the path does not want to change any more, we can stop

    # print best_path, city
    return best_path, dist

        
if __name__=='__main__':

    n_nodes = 20        # Number of cities to visit
    # Choosing city coordinates
    R=[]  # coordinates of cities are choosen randomly
    for i in range(n_nodes):
        R.append( [rand(),rand()] )
    R = array(R)

    # R = np.loadtxt('dump/12.pos')
    n_nodes = R.shape[0]

    edge_matrix = np.zeros(shape=(n_nodes, n_nodes))

    # The index table -- the order the cities are visited.
    city = range(n_nodes)
    # Distance of the travel at the beginning
    dist = Total_Distance(city, edge_matrix)

    Plot(city, R, dist)
    # form a edge matrix for later use
    for i in range(n_nodes):
        for j in range(n_nodes):
            edge_matrix[i,j] = Distance(R[i], R[j])

    # edge_matrix = np.loadtxt('dump/12.dist')
    # np.savetxt('dump/12_0.dist', edge_matrix, fmt='%.6f')

    city, dist = solve(n_nodes, edge_matrix)

    Plot(city, R, dist)

