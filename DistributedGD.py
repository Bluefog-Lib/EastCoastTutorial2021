 
import bluefog.torch as bf
import torch

bf.init()

def generate_data(m, d):
    A = torch.randn(m, d).to(torch.double)
    ns = 0.1*torch.randn(m, 1).to(torch.double)
    x_o = torch.rand(d,1).to(torch.double)
    b = A.mm(x_o) + ns
    
    return A, b

def check_opt_cond(x, A, b):
    
    grad_local = A.t().mm(A.mm(x) - b)
    grad = bf.allreduce(grad_local, name='gradient')  # global gradient
    
    # the norm of global gradient is expected to be 0 (optimality condition)
    global_grad_norm = torch.norm(grad, p=2)
    print("[Distributed Grad Descent] Rank {}: global gradient norm: {}".format(bf.rank(), global_grad_norm))
        
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

    check_opt_cond(x_opt, A, b)
    
    return x_opt

m, d = 20, 5 # dimension of A
A, b = generate_data(m, d)
x_opt = distributed_grad_descent(A, b, maxite=200, alpha=1e-2)
