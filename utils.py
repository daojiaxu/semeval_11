import os
import sys
import logging

import ast
import pandas as pd
from collections import defaultdict

def load_train_file(filename):
    train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')

    texts = []; labels = []; label_tag_dict = {}; tag_label_dict = {}
    temp_dict = defaultdict(float)
    for i, text in enumerate(train_df['norm_text']):
        text = ast.literal_eval(text)
        texts.append(text)

        label = ast.literal_eval(train_df['labels'][i])

        for j in range(len(label)):
            temp_dict[label[j]] += 1

    for i, label in enumerate(temp_dict.keys()):
        label_tag_dict[label] = i
        tag_label_dict[i] = label

    for i, text in enumerate(train_df['norm_text']):
        label = ast.literal_eval(train_df['labels'][i])
        line_label = []

        for j in range(len(label)):
            line_label.append(label_tag_dict[label[j]])

        labels.append(line_label)

    print("text length: " + str(len(texts)))
    print("label length: " + str(len(labels)))
    print(temp_dict)
    print(label_tag_dict)
    print(tag_label_dict)

    return texts, labels, label_tag_dict, tag_label_dict


def load_test_file(filename):
    train_df = pd.read_csv(filename, header=0, sep=',', encoding='gbk')

    texts = []
    for i, text in enumerate(train_df['norm_text']):
        text = ast.literal_eval(text)
        texts.append(text)

    print("text length: " + str(len(texts)))

    return texts

