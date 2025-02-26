# 第一步：准备训练语料
sentences = [
    "我 喜欢 玩具",
    "我 爱 爸爸",
    "我 讨厌 挨打"
]
# 制作词汇表
word_list = " ".join(sentences).split()
print(f"[step1] - word_list(no Deduplication):{word_list}")
# 去重
word_list = (set(word_list))
print(f"[step1] - word_list(Deduplication):{word_list}")
# 制作索引表（word-index and index-word）
idx_to_word = {idx: word for idx, word in enumerate(word_list)}
print(f"[step1] - idx_to_word:{idx_to_word}")
word_to_idx = {word: idx for idx, word in enumerate(word_list)}
print(f"[step1] - word_to_idx:{word_to_idx}")

# 第二步：根据语料库生成训练数据（正式开始使用pytorch进行数据预处理）
import torch
import random

batch_size = 2  # 每批数据的大小


def make_batch():
    input_batch = []  # 定义输入批处理列表
    target_batch = []  # 定义目标批处理列表
    # 随机获取一批句子
    selected_sentences = random.sample(sentences, batch_size)
    # 制作输入批处理和对应的目标批处理
    for sentence in selected_sentences:
        words = sentence.split()
        input = [word_to_idx[word] for word in words[:-1]]
        target = word_to_idx[words[-1]]
        input_batch.append(input)
        target_batch.append(target)
    # 将自然语言转化为对应可输入模型的张量
    input_batch = torch.LongTensor(input_batch)  # 将数组转变为Long类型的Tensor，维度相同
    target_batch = torch.LongTensor(target_batch)
    return input_batch, target_batch


# 进行训练数据的生成(数据预处理)
input_batch, target_batch = make_batch()
print(f"[step2] - input_batch:{input_batch}")
print(f"[step2] - target_batch:{target_batch}")

# 第三步：定义NPLM模型类
import torch.nn as nn

n_step = 2
n_hidden = 2
voc_size = len(word_list)
embedding_size = 2


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.C = nn.Embedding(voc_size, embedding_size)  # 第一个词嵌入层
        self.lstm = nn.LSTM(embedding_size, n_hidden, batch_first=True)
        self.linear = nn.Linear(n_hidden, voc_size)

    def forward(self, X):
        X = self.C(X)
        lstm_out,_ = self.lstm(X)
        # 只选择最后一个时间步的输出作为全链接层的输入，通过第二个线性层得到输出
        output = self.linear(lstm_out[:,-1,:])
        return output


# 第四步：实例话NPLM模型类
model = LSTM()
print(f"[step4] - 模型结构：{model}")
# 第五步：训练
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
# 训练模型
for epoch in range(5000):
    optimizer.zero_grad()
    input_batch, target_batch = make_batch()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print('loss:','{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# 第六步：预测
input_strs = [['我','讨厌'],['我','喜欢']]

input_indices = [[ word_to_idx[word] for word in seq] for seq in input_strs] # 转变为对应下标

input_batch = torch.LongTensor(input_indices) # 变化为张量

predict = model(input_batch).data.max(1)[1] # 进行预测，并取输出中概率最大的类型

# 转化为对应的词
predict_strs =[ idx_to_word[n.item()] for n in predict.squeeze()]
for input_str,pred in zip(input_strs, predict_strs):
    print(f"[step5] - {input_str}:{pred}")