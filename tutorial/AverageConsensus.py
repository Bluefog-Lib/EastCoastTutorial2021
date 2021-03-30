
import os
import numpy as np
import bluefog.torch as bf
import torch
from bluefog.common import topology_util
import scipy.io as sio

bf.init()

def avg_consensus_one_step(x, x_global_average):
    # one-step average consensus. 
    x_next = bf.neighbor_allreduce(x)
    
    # the relative error: |x^k-x_gloval_average|/|x_gloval_average|
    rel_error = torch.norm(x_next-x_global_average, p=2)/torch.norm(x_global_average,p=2)

    return x_next, rel_error

# average consensus starts
d = 10
maxite = 60
torch.manual_seed(12345 * bf.rank())
x0 = torch.randn((d, 1)).to(torch.double)
x_global_average = bf.allreduce(x0)
rel_error_dict = {}

for graph in ['Ring', 'Mesh', 'Exp2']:
    
    if bf.rank() == 1:
        print('Runing average consensus with topology {}'.format(graph))
    
    rel_error = torch.zeros((maxite, 1))
    if graph == 'Ring':
        G = topology_util.RingGraph(bf.size())
    elif graph == 'Mesh':
        G = topology_util.MeshGrid2DGraph(bf.size())
    elif graph == 'Exp2':
        G = topology_util.ExponentialTwoGraph(bf.size())

    assert bf.set_topology(G, is_weighted=True)
        
    # average consensus
    x = x0.clone()
    for ite in range(maxite):

        if bf.rank()==0:
            if ite%10 == 0:
                print('Progress {}/{}'.format(ite, maxite))

        x, rel_error[ite] = avg_consensus_one_step(x, x_global_average)

    # save data 
    rel_err_np = rel_error.cpu().detach().numpy()
    rel_error_dict[graph] = rel_err_np

if bf.rank() == 0:
    result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'aveCns.mat')
    sio.savemat(result_file, rel_error_dict)
