# 第一步：准备语料库
datasets = [
    "我喜欢吃苹果",
    "我喜欢吃香蕉",
    "我喜欢吃葡萄",
    "我不喜欢吃香蕉",
    "他喜欢吃苹果",
    "她喜欢吃草莓",
]


# 第二步：分词准备
def tokenizer(text):
    """ 将字符串分词成为每一个单独的字符 """
    return [char for char in text]


# 第三步：根据输入的N，去进行字符元祖的划分（将句子分成N个Gram）
# 这一步要注意N很关键，N制定了Gram元组的长度，以及包含上下文的长度N-1，
# 比如你要划分我喜欢吃苹果，N=2，那么就要按照2的长度去划分元组，并且每次移动的步长为1
N = 2
# 准备一个空元组列表
gram_list = []
for text in datasets:
    chars = tokenizer(text)
    gram = []
    for index in range(0, len(chars), N-1):
        starChat = chars[index]
        if index + 1 >= len(chars):
            endChar = "."
        else:
            endChar = chars[index + 1]
        gram.append(starChat + endChar)
    gram_list.append(gram)
print(gram_list)

