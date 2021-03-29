 
import bluefog.torch as bf
import torch

bf.init()
x_local = torch.ones(1) * bf.rank()
x_bar = bf.allreduce(x_local)
print('Node {} achieved the global average {}'.format(bf.rank(), x_bar[0]))
