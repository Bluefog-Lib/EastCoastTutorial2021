import numpy as np
import bluefog.torch as bf
import torch
from bluefog.common import topology_util
import networkx as nx
import matplotlib.pyplot as plt

# Generate different A and b through different seed.
bf.init()
bf.set_topology(topology_util.RingGraph(bf.size()))
torch.manual_seed(12345 * bf.rank())

# Problem Size
M, N = 25, 10


def LossL2(A, b, x):
    # f_i(x) = \frac{1}{2} \|Ax-b\|^2
    return torch.norm(A.mm(x) - b) / 2


def GradientL2(A, b, x):
    # f_i(x) = A^T(Ax-b)
    return A.t().mm(A.mm(x) - b)


def ProxL2Step(A, b, v, n_i, alpha):
    m = A.shape[1]
    AA_inv = (A.t().mm(A) + alpha * n_i * torch.eye(m)).inverse()
    return AA_inv.mm(A.t().mm(b) + alpha * n_i * v)


def DecenADMMStepL2(A, b, x, a, v, n_i, alpha):
    next_x = ProxL2Step(A, b, v, n_i, alpha)
    neighbor_weights = {r: 0.5 / n_i for r in bf.in_neighbor_ranks()}
    next_a = bf.neighbor_allreduce(
        next_x, self_weight=0.5, neighbor_weights=neighbor_weights
    )
    next_v = v + next_a - a
    return next_x, next_a, next_v


def AllreduceGradient(A, b):
    x = torch.zeros(N, 1).to(torch.double)
    x = bf.broadcast(x, root_rank=0)  # make everyone starts from same point
    alpha = 0.01
    loss_records = []
    with torch.no_grad():
        for i in range(200):
            global_grad = bf.allreduce(GradientL2(A, b, x))
            x = (x - alpha * global_grad).clone()
            loss = bf.allreduce(LossL2(A, b, x))
            loss_records.append(loss)
    if bf.rank() == 0:
        print(f"Allreduce {global_grad}")
    return x, loss_records


def DecenADMMAlgorithm(A, b):
    x = torch.zeros(N, 1).to(torch.double)
    a = torch.zeros(N, 1).to(torch.double)
    v = torch.zeros(N, 1).to(torch.double)
    alpha = 0.1
    n_i = len(bf.in_neighbor_ranks())

    loss_records = [bf.allreduce(LossL2(A, b, x))]
    grad_records = [bf.allreduce(GradientL2(A, b, x))]
    with torch.no_grad():
        for i in range(200):
            next_x, next_a, next_v = DecenADMMStepL2(A, b, x, a, v, n_i, alpha)
            x, a, v = next_x.clone(), next_a.clone(), next_v.clone()

            loss_records.append(bf.allreduce(LossL2(A, b, x)))
            grad_records.append(bf.allreduce(GradientL2(A, b, x)))

    if bf.rank() == 0:
        print(f"ADMM: {grad_records[-1]}")
    return x, loss_records, grad_records


if __name__ == "__main__":
    # Random function
    A = torch.randn(M, N).to(torch.double)
    b = torch.rand(M, 1).to(torch.double)

    x_ar, loss_records_ar = AllreduceGradient(A, b)
    x_admm, loss_records_admm, grad_records = DecenADMMAlgorithm(A, b)
    if bf.rank() == 0:
        print(f"x_ar: {x_ar}, x_admm: {x_admm}")
        plt.plot(loss_records_admm, label="Decentralized ADMM")
        plt.plot(loss_records_ar, label="Allreduce Gradient")
        plt.legend()
        plt.show()
    bf.barrier()
