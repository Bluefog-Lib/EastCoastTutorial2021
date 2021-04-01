
import numpy as np
import bluefog.torch as bf
import torch
from bluefog.common import topology_util
import networkx as nx
np.set_printoptions(precision=3, suppress=True, linewidth=200)

bf.init()
network_size = bf.size()

G = topology_util.ExponentialTwoGraph(network_size)
W = np.zeros((network_size, network_size))

for rank in range(network_size):
    self_weight, neighbor_weights = topology_util.GetRecvWeights(G, rank)
    W[rank,rank] = self_weight
    for r, v in neighbor_weights.items():
        W[rank, r] = v
        
if bf.rank() == 0:
    print(W)
    print()
    print('The sum of each col is:', np.sum(W,axis=0))
    print('The sum of each row is:', np.sum(W,axis=1))
    if np.sum(W, axis=1).all() and np.sum(W, axis=0).all():
        print('W is doubly stochastic.')
        
    w, _ = np.linalg.eig(W)
    w = np.abs(w)
    w_sorted = np.sort(w)
    print('The second largest eigenvalue is:', w_sorted[-2])
