# 第一步：准备数据集
from typing import List
import torch
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


# 第三步：制作未编码的训练集（由于Skip-Gram的核心在于通过目标词生成上下文相关，所以我们需要根据目标词去获取指定窗口大小下的上下文）
def build_train_set(word: List[str], window_size: int) -> List:
    """ 根据窗口大小进行目标字上下文获取
     :return [(word1,word2),(word3,word4),......]
     """
    dataset = []
    for sentence in sentences:
        # 将这个句子拆分为多个字
        sentence = sentence.split()
        # 遍历这些字
        for idx, word in enumerate(sentence):
            # 获取窗口内除他之外的相关字和目标字元组
            for neighbor in sentence[max(idx - window_size, 0): min(idx + window_size, len(sentence))]:
                if neighbor != word:
                    dataset.append((word, neighbor))
    return dataset


datasets = build_train_set(words, 2)
print("明文训练集：", datasets)

# 第三步：明文训练集要转化为统一张亮大小的输入格式（One-Hot编码）
# 制作一下word与词汇表中对应位置索引值的元组
word_to_idx = {word:idx for idx, word in enumerate(words)}
idx_to_word = {idx:word for idx, word in enumerate(words)}

def plaintext_to_onehot(word, word_to_idx):
    """ 将明文字符串转化为对应长度和形状的张量 """
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1
    return tensor

def find_idx_by_word(word, idx_to_word):
    return idx_to_word[word]

# onehot_datasets = [(plaintext_to_onehot(context,word_to_idx),word_to_idx[target]) for context,target in datasets]
# print(onehot_datasets)

# 编写Skip-Gram-Model类
import torch.nn as nn
class SkipGram(nn.Module):
    def __init__(self,voc_size,embedding_size):
        super(SkipGram,self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self,x):
        hidden = self.input_to_hidden(x)
        output = self.hidden_to_output(hidden)
        return output

embedding_size = 2
skip_gram = SkipGram(voc_size=len(words),embedding_size=embedding_size)

# 第四步：建立训练
learning_rate = 0.001
epochs = 1000
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(skip_gram.parameters(),lr=learning_rate)
# 开始训练循环
loss_value = []
for epoch in range(epochs):
    loss_sum = 0
    for center_word, context in datasets:
        X = plaintext_to_onehot(center_word,word_to_idx).float().unsqueeze(0)
        y_true = torch.tensor([word_to_idx[context]],dtype=torch.long)
        y_pred = skip_gram(X)
        loss = criterion(y_pred, y_true)
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch:{epoch+1},Loss:{loss_sum / len(datasets)}")
        loss_value.append(loss_sum / len(datasets))


# 嵌入结果展示
for word,idx in word_to_idx.items():
    print(f"{word}的向量表示:{skip_gram.input_to_hidden.weight[:,idx].detach().numpy()}")