import torch
import torch.nn.functional as F

x = torch.randn(2, 3, 4)

num_heads = 2
head_dim = 2

linear_q = torch.nn.Linear(4, 4)
linear_k = torch.nn.Linear(4, 4)
linear_v = torch.nn.Linear(4, 4)

Q = linear_q(x)
K = linear_k(x)
V = linear_v(x)


def split_heads(tensor, num_heads):
    batch_size, seq_len, feature_dim = tensor.size()
    head_dim = feature_dim // num_heads
    output = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    return output


Q = split_heads(Q, num_heads)
K = split_heads(K, num_heads)
V = split_heads(V, num_heads)

raw_weights = torch.matmul(Q, K.transpose(-2, -1))

scale_factor = K.size(-1) ** 0.5
scale_weights = raw_weights / scale_factor
atten_weights = F.softmax(scale_weights, dim=-1)

attn_output = torch.matmul(atten_weights, V)


# 拼接
def combine_heads(tensor):
    batch_size, num_heads, seq_len, head_dim = tensor.size()
    feature_dim = head_dim * num_heads
    output = tensor.transpose(1,2).contiguous().view(batch_size, seq_len, feature_dim)
    return output

attn_output = combine_heads(attn_output)
linear_out = torch.nn.Linear(4,4)
attn_output = linear_out(attn_output)
print(f"加权信息：{attn_output}")
