#!/usr/bin/env python
# coding: utf-8

# In[52]:


import os
from nltk import tokenize
import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


# In[53]:


# 文本
def get_labels_vector_new():
    texts = []
    list = os.listdir("train_dev_articles")
    for i in range(0, len(list)):
        f = open("train_dev_articles/" + list[i], encoding='utf8')
        texts.append(f.read())




    labels_test = []
    f = open("train_dev_labels.txt", encoding='utf8')
    labels_test.append(f.read())


    # In[56]:


    labels_test = labels_test[0].split('\n')


    # In[57]:


    list_labels = os.listdir("train_dev_articles")


    # In[59]:


    labels_tag = {}  # 存储每篇文章的分词
    for i in range(0, 446):
        labels_tag[list_labels[i][7:16]] = []

    for i in range(0, len(texts)):
        labels_tag[list_labels[i][7:16]].append(tokenizer.tokenize(texts[i]))

    labels_tag_qujian = {}  # 存储每篇文章的重点区间
    for i in range(0, len(list_labels)):
        labels_tag_qujian[list_labels[i][7:16]] = []


    # In[60]:


    labels_test[0].split('\t')


    # In[61]:


    file_num =len(labels_test)


    # In[62]:


    len(labels_test)


    # In[64]:


    for i in range(0,file_num-1):
        labels_tag_qujian[labels_test[i].split('\t')[0]].append(labels_test[i].split('\t')[1])
        labels_tag_qujian[labels_test[i].split('\t')[0]].append(labels_test[i].split('\t')[2])


    # In[65]:


    # 判断开始和结束区间
    labels_tag_word_qujian = {}  # 存储每篇文章的分词
    for i in range(0, len(list_labels)):
        labels_tag_word_qujian[list_labels[i][7:16]] = []
    i = 0
    k = 0
    # print(labels_tag[list_labels[1][7:16]])
    #  词标签转换
    for j in range(0, len(list_labels)):
        i = 0
        k = 0
        while (i < len(labels_tag_qujian[list_labels[j][7:16]])):
            start = labels_tag_qujian[list_labels[j][7:16]][i]
            end = labels_tag_qujian[list_labels[j][7:16]][i + 1]
            a = len(tokenizer.tokenize(texts[j][:int(start)]))  # 起始区间
            b = len(tokenizer.tokenize(texts[j][int(start):int(end)]))  # 范围
            c = a + b  # 终点区间
            labels_tag_word_qujian[list_labels[j][7:16]].append(a)
            labels_tag_word_qujian[list_labels[j][7:16]].append(c)
            k = k + 1
            i = 2 * k


    # In[ ]:


    print("--------------show--------")
    # print(labels_tag_word_qujian)
    #  标签向量
    labels_tag_word_vector = {}
    for i in range(0, len(list_labels)):
        labels_tag_word_vector[list_labels[i][7:16]] = []

    for j in range(0, len(list_labels)):
        for length in range(0, len(tokenizer.tokenize(texts[j]))):
            labels_tag_word_vector[list_labels[j][7:16]].append(0)

    for j in range(0, len(list_labels)):
        i = 0
        k = 0
        while (i < len(labels_tag_word_qujian[list_labels[j][7:16]])):
            start = labels_tag_word_qujian[list_labels[j][7:16]][i]
            end = labels_tag_word_qujian[list_labels[j][7:16]][i + 1]
            a = len(tokenizer.tokenize(texts[j]))
            if (int(start) < a and int(end) + 1 < a):
                for length in range(int(start), int(end) + 1):
                    labels_tag_word_vector[list_labels[j][7:16]][length] = 1
            k = k + 1
            i = 2 * k

            # print(labels_tag_word_vector)
    sum = 0
    for j in range(0, len(list_labels)):
        sum += len(labels_tag_word_vector[list_labels[j][7:16]])
    print(sum / len(list_labels))
    return labels_tag_word_vector




