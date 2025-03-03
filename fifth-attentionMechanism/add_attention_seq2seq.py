# 第一步：准备语料
from os import makedev

sentences = [
    ['咖哥 喜欢 小冰', '<sos> KaGe likes XiaoBing', 'KaGe likes XiaoBing <eos>'],
    ['我 爱 学习 人工智能', '<sos> I love studying AI', 'I love studying AI <eos>'],
    ['深度学习 改变 世界', '<sos> DL changed the world', 'DL changed the world <eos>'],
    ['自然 语言 处理 很 强大', '<sos> NLP is so powerful', 'NLP is so powerful <eos>'],
    ['神经网络 非常 复杂', '<sos> Neural-Nets are complex', 'Neural-Nets are complex <eos>']
]

# 进行词汇表的制作
word_list_cn, word_list_en = [], []

for s in sentences:
    word_list_cn.extend(s[0].split())
    word_list_en.extend(s[1].split())
    word_list_en.extend(s[2].split())
# 去重
word_list_cn = list(set(word_list_cn))
word_list_en = list(set(word_list_en))
# 构建单词到索引的映射
word2idx_cn = {word: idx for idx, word in enumerate(word_list_cn)}
word2idx_en = {word: idx for idx, word in enumerate(word_list_en)}
# 构建索引到单词的映射
idx2word_cn = {idx: word for idx, word in enumerate(word_list_cn)}
idx2word_en = {idx: word for idx, word in enumerate(word_list_en)}
# 计算词汇表的大小
voc_size_cn = len(idx2word_cn)
voc_size_en = len(idx2word_en)
print("句子大小：", len(sentences))
print("中文词汇表大小", len(word_list_cn))
print("英文词汇表大小", len(word_list_en))
print("中文词汇索引", word2idx_cn)
print("英文词汇索引", idx2word_en)

# 第二步：生成Seq2Seq训练数据
import numpy as np
import torch
import random


# 定义一个函数，随机选择一个句子和词汇表生成输入，输出和目标数据
def make_data(sentences):
    # 随机选择一个句子进行训练
    random_sentence = random.choice(sentences)
    # 将输入句子中的单词转化为对应的索引
    encoder_input = np.array([[word2idx_cn[n] for n in random_sentence[0].split()]])
    # print(encoder_input)
    # 将输出句子中的单词转化为对应索引
    decoder_input = np.array([[word2idx_en[n] for n in random_sentence[1].split()]])
    # print(decoder_input)
    # 将目标句子中的单词转化为对应索引
    target_input = np.array([[word2idx_en[n] for n in random_sentence[2].split()]])
    # print(target_input)
    # 将输入、输出、目标批次转化为张量
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    target_input = torch.LongTensor(target_input)
    return encoder_input, decoder_input, target_input


encoder_input, decoder_input, target_input = make_data(sentences)
for s in sentences:  # 找出选的是那个句子
    if all([word2idx_cn[w] in encoder_input[0] for w in s[0].split()]):
        original_sentence = s
        break

print(f"选中original_sentence:{original_sentence}")
print(f"编码器输入张量的形状:{encoder_input.shape}")
print(f"解码器输入张量的形状:{decoder_input.shape}")
print(f"目标张量的形状:{target_input.shape}")
print(f"编码器输入张量:{encoder_input}")
print(f"解码器输入张量:{decoder_input}")
print(f"目标张量:{target_input}")

# 新增内容：定义注意力机制(这个主要是通过让解码器中每一时间步（x1）去注意编码器中的（x2），从而辅助生成更贴合上下文的序列)
# 所以改造的重点放在解码器中
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, decoder_context, encoder_context):
        # 计算点积
        raw_weights = torch.bmm(decoder_context, encoder_context.transpose(-2, -1))

        # 进行归一化
        atten_weights = F.softmax(raw_weights, dim=-1)

        # 进行加权相加
        atten_output = torch.bmm(atten_weights, encoder_context)
        return atten_output, raw_weights


# 第三步：定义编码器和解码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs)
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden


# 新增注意力机制
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.attention = Attention()  # 新增注意力层
        self.out = nn.Linear(2 * hidden_size, output_size)  # 新增考虑上下文向量2*hidden_size

    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs)
        output, hidden = self.rnn(embedded, hidden)
        context, atten_weights = self.attention(output, encoder_outputs)
        dec_output = torch.cat((output, context), dim=-1)
        dec_output = self.out(dec_output)
        return dec_output, hidden, atten_weights


n_hidden = 128

# 创建编码器
encoder = Encoder(voc_size_cn, n_hidden)
decoder = Decoder(n_hidden, voc_size_en)
print(f"编码器：{encoder}")
print(f"解码器:{decoder}")


# 第四步：定义seq2seq架构(注意力改造)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        # 初始化编码器和解码器
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input, hidden, dec_input):
        encoder_output, encoder_hidden = self.encoder(enc_input, hidden)
        decoder_hidden = encoder_hidden
        decoder_output, _, atten_weights = self.decoder(dec_input, decoder_hidden, encoder_output)  # 新增参数，编码器的输出
        return decoder_output, atten_weights


model = Seq2Seq(encoder, decoder)
print(f"Seq2Seq模型结构:{model}")


# 第五步：训练Seq2Seq架构
def train_seq2seq(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        encoder_input, decoder_input, target = make_data(sentences)
        hidden = torch.zeros(1, encoder_input.size(0), n_hidden)  # 初始中间态
        optimizer.zero_grad()
        decoder_output = model(encoder_input, hidden, decoder_input)
        loss = criterion(decoder_output.view(-1, voc_size_en), target.view(-1))
        if (epoch + 1) % 40 == 0:
            print(f"Epoch:{epoch + 1:04d} cost = {loss:.6f}")
        loss.backward()
        optimizer.step()


# 训练
epochs = 400
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_seq2seq(model, criterion, optimizer, epochs)


# 第六步：测试
def test_seq2seq(model, source_sentence):
    encoder_input = np.array([[word2idx_cn[n] for n in source_sentence.split()]])
    decoder_input = np.array([word2idx_en['<sos>']] + [word2idx_en['<eos>']] * (len(encoder_input[0]) - 1))
    # 转换LongTensor类型
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input).unsqueeze(0)
    hidden = torch.zeros(1, encoder_input.size(0), n_hidden)
    decoder_output, _ = model(encoder_input, hidden, decoder_input)  # 只取 decoder_output
    predict = decoder_output.data.max(2, keepdim=True)[1]
    # 打印输入的句子和预测的句子
    print(source_sentence, '->', [idx2word_en[n.item()] for n in predict.squeeze()])


# 测试模型
test_seq2seq(model, '咖哥 喜欢 小冰')
test_seq2seq(model, '自然 语言 处理 很 强大')
