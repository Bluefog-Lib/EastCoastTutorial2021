import os

import bluefog.torch as bf
import matplotlib.pyplot as plt
import torch

# Generate different A and b through different seed.
bf.init()
torch.manual_seed(12345 * bf.rank())


def generate_data(m, d):
    A = torch.randn(m, d).to(torch.double)
    ns = 0.1 * torch.randn(m, 1).to(torch.double)
    x_o = torch.rand(d, 1).to(torch.double)
    b = A.mm(x_o) + ns

    return A, b


def LossL2(A, b, x):
    # f_i(x) = \frac{1}{2} \|Ax-b\|^2
    return torch.norm(A.mm(x) - b) / 2


def GradientL2(A, b, x):
    # f_i(x) = A^T(Ax-b)
    return A.t().mm(A.mm(x) - b)


def ProximalStepL2(A, b, x, y, u, alpha):
    # x^+ = argmin( 0.5*\|Ax-b\|^2 + u^t (x-y) + \alpha/2*\|x-y\|^2)
    #     = {(find x s.t.) A^t(Ax-b) + u + \alpha (x - y) = 0}
    #     = (A^tA + \alpha I)^{-1} (\alpha y + A^tb - u )
    m = A.shape[1]
    AA_inv = (A.t().mm(A) + alpha * torch.eye(m)).inverse()
    return AA_inv.mm(alpha * y + A.t().mm(b) - u)


# Ex: Make corresponding steps for l1 + l2 cases.


def CentralizedADMMStepL2(A, b, x, y, u, alpha):
    next_x = ProximalStepL2(A, b, x, y, u, alpha)
    # We use allreduce to mimic the centralized behavior
    # It should be based on PS architecture and using gather and broadcast.
    next_y = bf.allreduce(next_x)  # Without u is okay since allreudce(u) == 0
    next_u = u + alpha * (next_x - next_y)
    return next_x, next_y, next_u


def AllreduceGradient(A, b):
    d = A.shape[1]
    x = torch.zeros(d, 1).to(torch.double)
    x = bf.broadcast(x, root_rank=0)  # make everyone starts from same point
    mu = 0.01
    loss_records = []
    with torch.no_grad():
        for i in range(100):
            global_grad = bf.allreduce(GradientL2(A, b, x))
            x = (x - mu * global_grad).clone()
            loss = bf.allreduce(LossL2(A, b, x))
            loss_records.append(loss)
    return x, loss_records


def CentralizedADMMAlgorithm(A, b):
    d = A.shape[1]
    x = torch.zeros(d, 1).to(torch.double)
    y = torch.zeros(d, 1).to(torch.double)
    u = torch.zeros(d, 1).to(torch.double)
    alpha = 100
    loss_records = [bf.allreduce(LossL2(A, b, x))]
    with torch.no_grad():
        for i in range(100):
            next_x, next_y, next_u = CentralizedADMMStepL2(A, b, x, y, u, alpha)
            x, y, u = next_x.clone(), next_y.clone(), next_u.clone()

            loss_records.append(bf.allreduce(LossL2(A, b, y)))

    global_grad = bf.allreduce(GradientL2(A, b, x))
    print(
        f"[Centralized ADMM] Rank {bf.rank()}: ADMM residue gradient norm: " +
        f"{torch.norm(global_grad) / len(global_grad)}"
    )
    return x, loss_records


if __name__ == "__main__":
    # Problem Size
    A, b = generate_data(m=100, d=50)

    x_ar, loss_records_ar = AllreduceGradient(A, b)
    x_admm, loss_records_admm = CentralizedADMMAlgorithm(A, b)
    x_admm_ar = bf.allreduce(x_admm)
    if bf.rank() == 0:
        print(f"Last three entries of x_ar:\n {x_ar[-3:]}")
        print(f"Last three entries of x_admm:\n {x_admm_ar[-3:]}")
        plt.plot(loss_records_admm, label="Centralized ADMM")
        plt.plot(loss_records_ar, label="Allreduce Gradient")
        plt.legend()
        dirname = 'images'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(os.path.join(dirname, 'centralized_admm.png'))

    bf.barrier()
