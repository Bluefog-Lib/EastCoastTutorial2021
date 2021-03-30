 
import bluefog.torch as bf
import torch

bf.init()
# Make sure different agent has different random seed.
torch.manual_seed(12345 * bf.rank())

def generate_data(m, d):
    A = torch.randn(m, d).to(torch.double)
    B_inv = torch.randn(d, d).to(torch.double)
    ns = 0.1*torch.randn(m, 1).to(torch.double)
    x_o = torch.rand(d,1).to(torch.double)
    b = A.mm(x_o) + ns
    
    return A, B_inv, b

def soft_threshold(x, kappa):
    
    zeros = torch.zeros(d,1).to(torch.double)
    x = torch.max(x - kappa, zeros)
    x = torch.min(x + kappa, zeros)
    
    return x

def check_opt_cond(y, A, B_inv, b, alpha):
    
    m, d = A.shape
    n = bf.size()
    C = A.mm(B_inv)
    
    # update z
    grad_local = C.t().mm(C.mm(y) - b)
    z = y - alpha*grad_local
    z_bar = bf.allreduce(z)
    z_sum = z_bar * n

    # update v
    v = (1/(n*alpha*alpha)) * (alpha*z_sum - soft_threshold(alpha*z_sum, n*alpha*alpha))

    # update y
    y_next = z - alpha*v
    
    # the norm of global gradient is expected to be 0 (optimality condition)
    global_grad_norm = torch.norm((y - y_next), p=2)
    print("[Primal Decomposition] Rank {}: optimality metric norm: {}".format(bf.rank(), global_grad_norm))
        
    return

def primal_decomposition(A, B_inv, b, maxite=5000, alpha=1e-1):

    m, d = A.shape
    n = bf.size()
    C = A.mm(B_inv)
    
    y = torch.zeros(d, 1, dtype=torch.double)

    for _ in range(maxite):
        # update z
        grad_local = C.t().mm(C.mm(y) - b)
        z = y - alpha*grad_local
        z_bar = bf.allreduce(z)
        z_sum = z_bar * n
        
        # update v
        v = (1/(n*alpha*alpha)) * (alpha*z_sum - soft_threshold(alpha*z_sum, n*alpha*alpha))
        
        # update y
        y = z - alpha*v

        # update x
        x = B_inv.mm(y)

    check_opt_cond(y, A, B_inv, b, alpha)
    
    return y

if __name__ == "__main__":
    m, d = 20, 5 # dimension of A
    A, B_inv, b = generate_data(m, d)
    y = primal_decomposition(A, B_inv, b, maxite=10000, alpha=3e-3)
