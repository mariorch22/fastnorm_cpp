import torch
from torch.nn import RMSNorm
import fastnorm
import timeit

x = torch.randn(32, 4096, 768)

# PyTorch
layer = RMSNorm(768)
print(f"PyTorch: {timeit.timeit(lambda: layer(x), number=10) / 10}")

# FastNorm  
print(f"FastNorm: {timeit.timeit(lambda: fastnorm.rmsnorm(x), number=10) / 10}")