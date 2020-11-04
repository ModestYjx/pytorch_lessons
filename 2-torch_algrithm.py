import torch
import numpy as np
torch_data1 = torch.FloatTensor([0])
torch_data2 = torch.ones(5)
torch_data3 = torch.zeros(5)
print(torch_data1)
print(torch_data2)
print(torch_data3)
print(torch.add(torch_data1, torch_data2))
print(torch.sub(torch_data1, torch_data2))
print(torch.mul(torch_data1, torch_data2))
