# 需要缩放因子的原因是因为有些值的原始权重值过大，不利于归一化
# 准备两个输入序列张量
import torch
x1 = torch.randn(2,3,4)
x2 = torch.randn(2,5,4)

# 求原始权重
raw_weights = torch.bmm(x1,x2.transpose(1,2))

# 提取缩放因子（维度大小的平方根）
scaling_factor =  x1.size(-1)**0.5

# 计算出缩放权重矩阵
scaling_weights = raw_weights / scaling_factor

# 使用缩放权重矩阵进行归一化
import torch.nn.functional as F
atten_weights = F.softmax(scaling_weights,dim=-1)

# 对归一化之后的缩放权重矩阵进行加权相加处理
atten_output = torch.bmm(atten_weights,x2)

print(atten_output)

