# CBOW是通过周围词进行中心词预测的方式
# 第一步：准备数据集
import torch
import torch.nn as nn
from torch.nn.functional import embedding
import numpy as np
import torch.optim as optim

sentences = [
    "Kage is Teacher",
    "Mazong is Boss",
    "Niuzong is Boss",
    "Xiaobing is Student",
    "Xiaoxue is Student"
]

# 第二步：进行词汇表的制作
# 将数据集中的词进行提取
words = " ".join(sentences).split()
# 去重
words = list(set(words))
print("词汇表为：", words)

# 第三步：制作明文训练集((target,['context1','context2']))
# 通过滑动窗口的方式，相当于从每个词便利，上下窗口都是周围词，放入数组中即可
def create_cbow_dataset(sentences,window_size=2):
    data = []
    for sentence in sentences:
        # separate word from sentences
        sentence_words = sentence.split()
        # 遍历词，并且获取他窗口之内的周围词
        for idx, word in enumerate(sentence_words):
            context_words = (sentence_words[max(idx-window_size,0):idx]
                             + sentence_words[idx+1:min(idx+window_size+1,len(sentence_words))])
            data.append((word,context_words))
    return data


cbow_data = create_cbow_dataset(sentences=sentences,window_size=2)

# 第四步：定义明文转输入向量函数，明文训练集要转化为统一张亮大小的输入格式（One-Hot编码）
# 制作一下word与词汇表中对应位置索引值的元组
word_to_idx = {word:idx for idx, word in enumerate(words)}
idx_to_word = {idx:word for idx, word in enumerate(words)}

def plaintext_to_onehot(word, word_to_idx):
    """ 将明文字符串转化为对应长度和形状的张量 """
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1
    return tensor

# 第五步：定义模型
class CBOW(nn.Module):
    def __init__(self,voc_size,embedding_size):
        super(CBOW, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size,embedding_size,bias=False)
        self.hidden_to_output = nn.Linear(embedding_size,voc_size,bias=False)

    def forward(self,x):
        embedding = self.input_to_hidden(x)
        avg_embedding = torch.mean(embedding,dim=0)
        output = self.hidden_to_output(avg_embedding.unsqueeze(0))
        return output
embedding_size = 2
cbow_model = CBOW(voc_size=len(words),embedding_size=embedding_size)

#训练
# 第四步：建立训练
learning_rate = 0.001
epochs = 1000
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cbow_model.parameters(),lr=learning_rate)
# 开始训练循环
loss_value = []
for epoch in range(epochs):
    loss_sum = 0
    for target, context_words in cbow_data:
        X = torch.stack([plaintext_to_onehot(word,word_to_idx) for word in context_words]).float()
        y_true = torch.tensor([word_to_idx[target]],dtype=torch.long)
        y_pred = cbow_model(X)
        loss = criterion(y_pred, y_true)
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch:{epoch+1},Loss:{loss_sum / len(cbow_data)}")
        loss_value.append(loss_sum / len(cbow_data))
