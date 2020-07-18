from keras.models import load_model
# from semeval.datasets import pre_deal
from transformers import BertTokenizer
import pre_deal
import pre_deal_bert
from bert_serving.client import BertClient
from nltk import tokenize
import os
from itertools import groupby
# from semeval.datasets import KMP
import KMP
import numpy as np
import random

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# list_tc = ['Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon,Reductio_ad_hitlerum',
# #            'Black-and-White_Fallacy',
# #            'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language',
# #            'Name_Calling,Labeling', 'Repetition', 'Slogans', 'Thought-terminating_Cliches',
# #            'Whataboutism,Straw_Men,Red_Herring']
max_len = 1000

class_TC = []
f = open("propaganda-techniques-names-semeval2020task11.txt", encoding='utf8')
class_TC.append(f.read())
s = class_TC[0]
class_TC = s.split('\n')
class_TC.remove("")
# bc = BertClient(ip='222.19.197.230', port=5555, port_out=5556, check_version=False)
model = load_model('sigmoid_mode_9.h5')
test_text = pre_deal_bert.get_test_textVector()
# test_vectors = bc.encode(test_text)
test_vectors = np.load("glove_test_300d.npy")
# np.save("test_case_512.npy", test_vectors)
test_predict_gailv = model.predict(test_vectors)
print("--------------show---------------")
test_pred = model.predict(test_vectors).argmax(-1)
print("test_pred")
print(test_pred)
print(test_pred.shape)
test_pred_jieguo = []
for i in range(0, 75):
    test_pred_jieguo.append([])

for i in range(0, 75):
    for j in range(0, max_len):
        if test_predict_gailv[i][j][1] > 0.285:
            test_pred_jieguo[i].append(1)
        else:
            test_pred_jieguo[i].append(0)

texts_token = []
for i in range(0, len(test_text)):
    texts_token.append(tokenizer.tokenize(test_text[i]))
# end = 0
filename = 'a.txt'
f = open(filename, 'w', encoding='utf-8')
labels = []
list_labels = os.listdir("dev-articles")
labels_tag = {}  # 存储每篇文章的分词
for i in range(0, len(list_labels)):
    labels_tag[list_labels[i][7:16]] = []

# 获得测试集的word区间
for j in range(0, 75):
    text_index = []
    for i in range(0, max_len):
        if (test_pred_jieguo[j][i] == 1):
            text_index.append(i)
    fun = lambda x: x[1] - x[0]
    for k, g in groupby(enumerate(text_index), fun):
        l1 = [j for i, j in g]  # 连续数字的列表
        if len(l1) > 1:
            scop = str(min(l1)) + '-' + str(max(l1))  # 将连续数字范围用"-"连接
        else:
            scop = l1[0]
        labels_tag[list_labels[j][7:16]].append(min(l1))
        labels_tag[list_labels[j][7:16]].append(max(l1))
        print("----------------------")
        if (min(l1) == max(l1) and min(l1) > 400):
            li = texts_token[j][min(l1):]
            print(texts_token[j][min(l1):])
        else:
            li = texts_token[j][min(l1):max(l1)]
            print(texts_token[j][min(l1):max(l1)])

        list2 = [str(i) for i in li]  # 使用列表推导式把列表中的单个元素全部转化为str类型
        list3 = ' '.join(list2)  # 把列表中的元素放在空串中，元素间用空格隔开
        if (list3 == ''):
            pass
        else:
            a = KMP.KMP_algorithm(test_text[j], list3)
            if (a == -1):
                list_gai = str(texts_token[j][min(l1)])
                a = KMP.KMP_algorithm(test_text[j], list_gai)  # 开始位置
                print(list_labels[j][7:16])
                print("值为：" + str(a))
                b = a + len(list3)
                print("结束值为：" + str(b))
                # str_1 = random.choice(list_tc)
                random.random()
                if (a == -1):
                    if (b > 100):
                        f.write(list_labels[j][7:16] + '\t' + class_TC[random.randint(0, 13)] + '\t' + str(
                            b - 100) + '\t' + str(b) + '\n')
                    else:
                        f.write(
                            list_labels[j][7:16] + '\t' + class_TC[random.randint(0, 13)] + '\t' + str(0) + '\t' + str(
                                b) + '\n')
                else:
                    f.write(list_labels[j][7:16] + '\t' + class_TC[random.randint(0, 13)] + '\t' + str(a) + '\t' + str(
                        b) + '\n')
            else:
                print(list_labels[j][7:16])
                print("值为：" + str(a))
                b = a + len(list3)
                print("结束值为：" + str(b))
                # str_1 = random.choice(list_tc)
                if (a == -1):
                    if (b > 100):
                        f.write(list_labels[j][7:16] + '\t' + class_TC[random.randint(0, 13)] + '\t' + str(
                            b - 100) + '\t' + str(b) + '\n')
                    else:
                        f.write(
                            list_labels[j][7:16] + '\t' + class_TC[random.randint(0, 13)] + '\t' + str(0) + '\t' + str(
                                b) + '\n')
                else:
                    f.write(list_labels[j][7:16] + '\t' + class_TC[random.randint(0, 13)] + '\t' + str(a) + '\t' + str(
                        b) + '\n')
            # a = KMP.KMP_algorithm(test_text[j],list3)
            # print("值为："+a)
        print(list3)
        print(str(min(l1)))
        print(str(max(l1)))
        print("连续数字范围：{}".format(scop))

f.close()
