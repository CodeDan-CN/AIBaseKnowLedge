# 准备两个相同形状的张量（batch_size,seq_len1,feature_dim）
import torch

x1 = torch.randn(2,3,4) #（batch_size,seq_len1,feature_dim）
x2 = torch.randn(2,5,4) #（batch_size,seq_len1,feature_dim）

print(f"x1:{x1}")
print(f"x2:{x2}")

# 计算点积
# batch_size标识批次大小，seq_len表示序列的长度，feature_dim通常标识词嵌入的维度
# 所以x1和x2的不同在于输入的序列长度不同,并且feature_dim的维度值一定要一致,从数学上解释就是点积的计算，需要列数相同
raw_weights = torch.bmm(x1,x2.transpose(1,2))
print(f"raw_weights:{raw_weights}")

# 进行归一化
import torch.nn.functional as F
atten_weights = F.softmax(raw_weights,dim=2)

# 直接格式化张量为小数点后两位
formatted_weights = [[["{:.2f}".format(val.item()) for val in row] for row in batch] for batch in atten_weights]

# 打印格式化后的结果
print("formatted_weights:")
for batch in formatted_weights:
    for row in batch:
        print(row)

# 进行加权相加
atten_output = torch.bmm(atten_weights,x2)
print(f"atten_output:{atten_output}")




