import torch

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = x.eq(y)
print(z)