# 第一步：准备语料库
datasets = [
    "我喜欢吃苹果",
    "我喜欢吃香蕉",
    "她喜欢吃葡萄",
    "他不喜欢吃香蕉",
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
grams_list = []
chars_list = []
for text in datasets:
    chars = tokenizer(text)
    chars_list.append(chars)
    gram = []
    for index in range(0, len(chars), N-1):
        starChat = chars[index]
        if index + 1 >= len(chars):
            endChar = "."
        else:
            endChar = chars[index + 1]
        gram.append(starChat + endChar)
    grams_list.append(gram)
print(grams_list)

# 第三步：制作两份词典（python代码方式的话，字典统计）
singe_word_dict = {}
for chars in chars_list:
    for char in chars:
        if char not in singe_word_dict:
            singe_word_dict[char] = 1
        else:
            singe_word_dict[char] += 1
print(singe_word_dict)

gram_word_dict= {}
for grams in grams_list:
    for gram in grams:
        if gram not in gram_word_dict:
            gram_word_dict[gram] = 1
        else:
            gram_word_dict[gram] += 1
print(gram_word_dict)

# 第四步：计算gram出现概率(结果为一个嵌套字典)
final_word_dict = {}
for word,count in gram_word_dict.items():
    char = word[0]
    if char not in singe_word_dict:
        continue
    char_count = singe_word_dict[char]
    result = round((count / char_count) * 100,0)
    if char not in final_word_dict:
        temp_dict = {word[1:]: result}
        final_word_dict[char] = temp_dict
    else:
        final_word_dict[char][word[1:]]= result
print(final_word_dict)


# 预测
star_char = "葡"
result_str = star_char
while True:
    # 获取句子的最后一个字
    chat = result_str[-1]
    if chat not in final_word_dict:
        break
    option_words = final_word_dict[chat]
    temp_word = ''
    temp_count = 0
    # 选择概率
    for word,count in option_words.items():
        if count > temp_count:
            temp_word = word
            temp_count = count
    result_str  = result_str + temp_word
print(result_str)
