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
# 'sigmoid_mode_11.h5' 0.37目前最优

max_len = 1200
# max_files_nums = 90  # 90 or 75
max_files_nums = 75
# file_name = "test-articles"
file_name = "dev-articles"
creat_txt_name = "glove_bilstm.txt"
# test_vectors_name = "glove_final_300d_1200.npy"
test_vectors_name = "glove_test_300d_1200.npy"
score = 0.42
model = load_model('glove_bilstm.h5')
test_text = pre_deal_bert.get_test_textVector()

test_vectors = np.load(test_vectors_name)

test_predict_gailv = model.predict(test_vectors)
print("--------------show---------------")
test_pred = model.predict(test_vectors).argmax(-1)
print("test_pred")
print(test_pred)
print(test_pred.shape)
test_pred_jieguo = []
for i in range(0, max_files_nums):
    test_pred_jieguo.append([])

for i in range(0, max_files_nums):
    for j in range(0, max_len):
        if test_predict_gailv[i][j][1] > score:
            test_pred_jieguo[i].append(1)
        else:
            test_pred_jieguo[i].append(0)

texts_token = []
for i in range(0, len(test_text)):
    texts_token.append(tokenizer.tokenize(test_text[i]))
# end = 0
filename = creat_txt_name
f = open(filename, 'w', encoding='utf-8')
labels = []
list_labels = os.listdir(file_name)
labels_tag = {}  # 存储每篇文章的分词
for i in range(0, len(list_labels)):
    labels_tag[list_labels[i][7:16]] = []

# 获得测试集的word区间
for j in range(0, max_files_nums):
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
            try:
                li = texts_token[j][min(l1):]
                print(texts_token[j][min(l1):])
            except IndexError:
                pass
        else:
            try:
                li = texts_token[j][min(l1):max(l1)]
                print(texts_token[j][min(l1):max(l1)])
            except IndexError:
                pass

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
                if (a == -1):
                    if (b > 100):
                        f.write(list_labels[j][7:16] + '\t' + str(b - 100) + '\t' + str(b) + '\n')
                    else:
                        f.write(list_labels[j][7:16] + '\t' + str(0) + '\t' + str(b) + '\n')
                else:
                    f.write(list_labels[j][7:16] + '\t' + str(a) + '\t' + str(b) + '\n')
            else:
                print(list_labels[j][7:16])
                print("值为：" + str(a))
                b = a + len(list3)
                print("结束值为：" + str(b))
                # str_1 = random.choice(list_tc)
                if (a == -1):
                    if (b > 100):
                        f.write(list_labels[j][7:16] + '\t' + str(b - 100) + '\t' + str(b) + '\n')
                    else:
                        f.write(list_labels[j][7:16] + '\t' + str(0) + '\t' + str(b) + '\n')
                else:
                    f.write(list_labels[j][7:16] + '\t' + str(a) + '\t' + str(b) + '\n')
            # a = KMP.KMP_algorithm(test_text[j],list3)
            # print("值为："+a)
        print(list3)
        print(str(min(l1)))
        print(str(max(l1)))
        print("连续数字范围：{}".format(scop))

f.close()
