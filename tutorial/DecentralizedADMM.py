import bluefog.torch as bf
from bluefog.common import topology_util
import matplotlib.pyplot as plt
import torch

# Generate different A and b through different seed.
bf.init()
bf.set_topology(topology_util.RingGraph(bf.size()))
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


def ProximalStepL2(A, b, v, n_i, alpha):
    # x^+ = argmin( 0.5*\|Ax-b\|^2 + \alpha |N_i| /2*\|x-v\|^2)
    #     = {(find x s.t.) A^t(Ax-b) + \alpha |N_i| v= 0}
    #     = (A^tA + \alpha I)^{-1} (\alpha |N_i| v)
    m = A.shape[1]
    AA_inv = (A.t().mm(A) + alpha * n_i * torch.eye(m)).inverse()
    return AA_inv.mm(A.t().mm(b) + alpha * n_i * v)


def DecentralizedADMMStepL2(A, b, x, a, v, n_i, alpha):
    next_x = ProximalStepL2(A, b, v, n_i, alpha)
    neighbor_weights = {r: 0.5 / n_i for r in bf.in_neighbor_ranks()}
    next_a = bf.neighbor_allreduce(
        next_x, self_weight=0.5, neighbor_weights=neighbor_weights
    )
    next_v = v + next_a - a
    return next_x, next_a, next_v


# Ex: Make corresponding steps for l1 + l2 cases.


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


def DecentralizedADMMAlgorithm(A, b):
    d = A.shape[1]
    x = torch.zeros(d, 1).to(torch.double)
    a = torch.zeros(d, 1).to(torch.double)
    v = torch.zeros(d, 1).to(torch.double)
    alpha = 100
    n_i = len(bf.in_neighbor_ranks())

    loss_records = [bf.allreduce(LossL2(A, b, x))]
    with torch.no_grad():
        for i in range(200):
            next_x, next_a, next_v = DecentralizedADMMStepL2(A, b, x, a, v, n_i, alpha)
            x, a, v = next_x.clone(), next_a.clone(), next_v.clone()

            loss_records.append(bf.allreduce(LossL2(A, b, x)))

    global_grad = bf.allreduce(GradientL2(A, b, x))
    print(
        f"[Decentralized ADMM] Rank {bf.rank()}: ADMM residue gradient norm: " +
        f"{torch.norm(global_grad) / len(global_grad)}"
    )
    return x, loss_records


if __name__ == "__main__":
    # Problem Size
    A, b = generate_data(m=100, d=50)

    x_ar, loss_records_ar = AllreduceGradient(A, b)
    x_admm, loss_records_admm = DecentralizedADMMAlgorithm(A, b)
    x_admm_ar = bf.allreduce(x_admm)
    if bf.rank() == 0:
        print(f"Last three entries of x_ar:\n {x_ar[-3:]}")
        print(f"Last three entries of x_admm:\n {x_admm_ar[-3:]}")
        # plt.plot(loss_records_admm, label="Decentralized ADMM")
        # plt.plot(loss_records_ar, label="Allreduce Gradient")
        # plt.legend()
        # plt.imsave()

    bf.barrier()
