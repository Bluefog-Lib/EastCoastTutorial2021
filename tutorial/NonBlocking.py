
import os
import numpy as np
import bluefog.torch as bf
import torch
from bluefog.common import topology_util
import networkx as nx
import scipy.io as sio
import time

bf.init()

def generate_data(m, d, x_o):
    A = torch.randn(m, d).to(torch.double)
    ns = 0.1*torch.randn(m, 1).to(torch.double)
    b = A.mm(x_o) + ns
    
    return A, b

def check_opt_cond(x, A, b):
    
    grad_local = A.t().mm(A.mm(x) - b)
    grad = bf.allreduce(grad_local, name='gradient')  # global gradient
    
    # the norm of global gradient is expected to be 0 (optimality condition)
    global_grad_norm = torch.norm(grad, p=2)
    if bf.rank() == 0:
        print("[Distributed Grad Descent] Rank {}: global gradient norm: {}".format(
            bf.rank(), global_grad_norm))
        
    return

def distributed_grad_descent(A, b, maxite=5000, alpha=1e-1):

    m, d = A.shape
    
    x_opt = torch.zeros(d, 1, dtype=torch.double)

    for _ in range(maxite):
        # calculate local gradient 
        grad_local = A.t().mm(A.mm(x_opt) - b)
        
        # global gradient
        grad = bf.allreduce(grad_local, name='gradient')

        # distributed gradient descent
        x_opt = x_opt - alpha*grad
    
    return x_opt

def ATC_DGD_one_step(x, x_opt, A, b, alpha=1e-2):
    
    # one-step ATC-DGD. 
    # The combination weights have been determined by the associated combination matrix.
    
    grad_local = A.t().mm(A.mm(x) - b)      # compute local grad
#     time.sleep(0.05)
    y = x - alpha*grad_local                # adapte
    x_new = bf.neighbor_allreduce(y)        # combination
    
    # the relative error: |x^k-x_gloval_average|/|x_gloval_average|
    rel_error = torch.norm(x_new-x_opt, p=2)/torch.norm(x_opt,p=2)

    return x_new, rel_error

def AWC_DGD_one_step(x, x_opt, A, b, alpha=1e-2):
    
    # one-step AWC-DGD. 
    # The combination weights have been determined by the associated combination matrix.
    
    grad_local = A.t().mm(A.mm(x) - b)                       # compute local grad
    x_new = bf.neighbor_allreduce(x) - alpha*grad_local      # AWC update
    
    # the relative error: |x^k-x_gloval_average|/|x_gloval_average|
    rel_error = torch.norm(x_new-x_opt, p=2)/torch.norm(x_opt,p=2)

    return x_new, rel_error

def NonBlocking_AWC_DGD_one_step(x, x_opt, A, b, alpha=1e-2):
    
    # one-step NBK-AWC-DGD. 
    # The combination weights have been determined by the associated combination matrix.
    
    x_handle = bf.neighbor_allreduce_nonblocking(x)
    grad_local = A.t().mm(A.mm(x) - b)                       # compute local grad
#     time.sleep(0.05)
    x_new = bf.synchronize(x_handle) - alpha*grad_local      # AWC update
    
    # the relative error: |x^k-x_gloval_average|/|x_gloval_average|
    rel_error = torch.norm(x_new-x_opt, p=2)/torch.norm(x_opt,p=2)

    return x_new, rel_error

if __name__ == "__main__":
    
    torch.manual_seed(12345 * bf.rank())
    
    m, d = 20, 2000 # dimension of A
    x_o = torch.rand(d,1).to(torch.double)
    x_o = bf.broadcast(x_o, root_rank = 0)
    A, b = generate_data(m, d, x_o)
    x_opt = distributed_grad_descent(A, b, maxite=200, alpha=1e-2)
    
    G = topology_util.ExponentialTwoGraph(bf.size())  # Set topology as exponential-two topology.
    bf.set_topology(G)

    maxite = 3000
    alpha = 5e-3
    rel_error_dict = {}
    
    for method in ['ATC', 'AWC', 'NBK-AWC']:
        
        start = time.time()
        
        if bf.rank() == 0:
            print('\nRunning {}:'.format(method))
    
        x = torch.zeros(d, 1, dtype=torch.double).to(torch.double)  # Initialize x
        rel_error = torch.zeros((maxite, 1))
        for ite in range(maxite):

            if bf.rank()==0:
                if ite%500 == 0:
                    print('Progress {}/{}'.format(ite, maxite))

            # you can adjust alpha to different values
            if method == 'ATC':
                x, rel_error[ite] = ATC_DGD_one_step(x, x_opt, A, b, alpha=alpha)
            elif method == 'NBK-AWC':
                x, rel_error[ite] = NonBlocking_AWC_DGD_one_step(x, x_opt, A, b, alpha=alpha)
            else:
                AWC_DGD_one_step(x, x_opt, A, b, alpha=alpha)
                
        rel_error_dict[method] = rel_error.cpu().detach().numpy().reshape(-1)
        
        end = time.time()
        
        if bf.rank() == 0:
            print('{} finishes in {} seconds.'.format(method, end - start))
        
    if bf.rank() == 0:
        result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'results', 'NonBlocking.mat')
        sio.savemat(result_file, rel_error_dict)
