# 第一步：整理语料库
datasets = [
    "我特别特别喜欢看电影",
    "这部电影真的是很好看的电影",
    "今天天气真好是难得的好天气",
    "我今天去看了一部电影",
    "电影院的电影都很好看"
]

# 第二步：使用jieba分词,通过jieba的cut方法可以获取到分词好的对象，通过将其转化为list列表即可获取到分词字符串列表
# print(list(jieba.cut(datasets[0])))
import jieba
# 制作词汇表
words_table = []
for dataset in datasets:
    words_table.append(list(jieba.cut(dataset)))
print(words_table)
# 遍历初始词汇表
final_words_table = {}
for words in words_table:
    for word in words:
        if word not in final_words_table:
            final_words_table[word] = len(final_words_table)
print(final_words_table)

# 第三步：将语料转化为向量
# 初始化向量表
row_number = len(words_table)
column_number = len(final_words_table)
# 初始化一个 rows x cols 的二维数组，所有元素为 0
handle_vector_datasets = [[0 for _ in range(column_number)] for _ in range(row_number)]
index = 0
for words in words_table:
    for word in words:
        if word in final_words_table:
            jndex = final_words_table[word]
            handle_vector_datasets[index][jndex] += 1
    index += 1

print(handle_vector_datasets)

# 第四步：预测
# 计算相似度(余炫)
import numpy as np

# 计算两个向量之间的余弦相似度
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)  # 向量的点积
    norm_A = np.linalg.norm(A)  # 向量A的模
    norm_B = np.linalg.norm(B)  # 向量B的模
    return dot_product / (norm_A * norm_B)

result = cosine_similarity(handle_vector_datasets[0], handle_vector_datasets[0])
print(result)