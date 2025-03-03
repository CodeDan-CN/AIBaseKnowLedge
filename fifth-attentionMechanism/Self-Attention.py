import torch

x = torch.randn(2,3,4)

linear_q = torch.nn.Linear(4,4)
linear_k = torch.nn.Linear(4,4)
linear_v = torch.nn.Linear(4,4)

Q = linear_q(x)
K = linear_k(x)
V = linear_v(x)

raw_weights = torch.bmm(Q,K.transpose(-2,-1))

scale_factor = K.size(-1)**0.5

scale_weights = raw_weights/scale_factor

atten_output = torch.bmm(scale_weights,V)

print(atten_output)