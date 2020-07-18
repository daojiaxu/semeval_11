import os
import sys
import logging

import numpy as np
import pandas as pd
import ast
import pickle
from keras.layers.core import Dense, Activation
import tensorflow as tf
# from bert_serving.client import BertClient
from keras.preprocessing import sequence

from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras_contrib.layers import CRF
import pre_deal
from transformers import BertTokenizer
from bert_serving.client import BertClient
from nltk import tokenize
from itertools import groupby
# from semeval.datasets import KMP
import KMP
from keras.layers import Embedding, Bidirectional, LSTM, Input, Masking, Dropout
from keras.models import Model
from keras.utils import to_categorical
import pre_deal_bert
from keras import optimizers
from bert_serving.client import BertClient


_EPSILON = 1e-7
hidden_dim = 256
batch_size = 25
# batch_size = 32
nb_epoch = 16
max_len = 1000


def create_model(maxlen, embedding_dim):
    sequence = Input(shape=(maxlen, embedding_dim,), dtype='float32')
    # Embedding(input_dim=(maxlen, embedding_dim,), output_dim=256,trainable=True, dtype='float32')(sequence)
    #embedded = Masking(mask_value=0.05)(sequence)

    #hidden1 = Bidirectional(LSTM(hidden_dim, return_sequences=True, recurrent_dropout=0.15))(embedded)
    # hidden2 = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True, recurrent_dropout=0.25))(hidden1)
    hidden1 = LSTM(100, return_sequences=True, recurrent_dropout=0.15)(sequence)
    # hidden2 = LSTM(hidden_dim//2, return_sequences=True, recurrent_dropout=0.1)(hidden1)
    Dense1 = Dropout(0.15)(hidden1)
    # Dense2 = Dense(64, activation='relu')(Dense1)
    # Dense3 = Dense(16, activation='relu')(Dense1)
    # Dense4 = Dense(8, activation='relu')(Dense1)
    crf = CRF(2, sparse_target=True, learn_mode='marginal')(Dense1)
    #output = Dense(2, activation='sigmoid')(Dense1)
    # output=AMSoftmax(3,3,0.35)(hidden)
    model = Model(inputs=sequence, outputs=crf)
    model.summary()

    return model

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    # labels_vector_dict, texts = pre_deal.get_labels_vector()
    # labels_vector = []
    # for key in labels_vector_dict.keys():
    #     labels_vector.append(labels_vector_dict[key])
    # labels_vector = sequence.pad_sequences(np.array(labels_vector), maxlen=max_len, padding='post')
    # labels_vector = to_categorical(np.asarray(labels_vector), num_classes=2)
    # bc = BertClient(ip='222.19.197.230', port=5555, port_out=5556, check_version=False)
    # texts_vector = bc.encode(texts)
    labels_vector = np.load("train_labels_vector_1000.npy")
    texts_vector = np.load("glove_300d.npy")
    labels_vector = labels_vector.reshape(371, max_len, 1)
    labels_vector = to_categorical(labels_vector)
    # keras model
    model = create_model(max_len, 300)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])
    model.compile('adam', loss="categorical_crossentropy", metrics=[crf_accuracy])
    model.fit(texts_vector, labels_vector, epochs=nb_epoch, batch_size=batch_size,
              validation_split=0.08)
    test_text = pre_deal.get_test_textVector()
    # test_vectors = bc.encode(test_text)
    test_vectors = np.load("glove_test_300d.npy")
    test_predict_gailv = model.predict(test_vectors)
    print("--------------show---------------")
    test_pred = model.predict(test_vectors).argmax(0)
    print("test_pred")
    print(test_pred)
    print(test_pred.shape)
    # model.save('sigmoid_mode_4.h5')
    # test_textTokenWord = pre_deal.get_test_textVector()
    # test_vectors = bc.encode(test_textTokenWord)
    # print(model.predict(test_vectors))
    # print("--------------show---------------")
    # test_pred = model.predict(test_vectors).argmax(-1)
    # print("test_pred")
    # print(test_pred)
    # print(test_pred.shape)
    test_pred_jieguo = []
    for i in range(0, 75):
        test_pred_jieguo.append([])
    # 调整概率
    for i in range(0, 75):
        for j in range(0, max_len):
            if test_predict_gailv[i][j][1] > 0.25:
                test_pred_jieguo[i].append(1)
            else:
                test_pred_jieguo[i].append(0)

    texts_token = []
    for i in range(0, len(test_text)):
        texts_token.append(tokenizer.tokenize(test_text[i]))
    # end = 0
    filename = 'b.txt'
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
                    f.write(list_labels[j][7:16] + '\t' + str(a) + '\t' + str(b) + '\n')
                else:
                    print(list_labels[j][7:16])
                    print("值为：" + str(a))
                    b = a + len(list3)
                    print("结束值为：" + str(b))
                    f.write(list_labels[j][7:16] + '\t' + str(a) + '\t' + str(b) + '\n')
                # a = KMP.KMP_algorithm(test_text[j],list3)
                # print("值为："+a)
            print(list3)
            print(str(min(l1)))
            print(str(max(l1)))
            print("连续数字范围：{}".format(scop))

    f.close()
