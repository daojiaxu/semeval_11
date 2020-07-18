import os

import numpy as np
# from semeval.datasets import pre_deal
import pre_deal_bert
import new_pre_deal
from keras.preprocessing import sequence
from mxnet.contrib import text
from transformers import BertTokenizer
import pandas as pd
from bert_serving.client import BertClient

max_len = 1000
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


def get_vector():
    # 文本
    texts = []
    list = os.listdir("train-articles")
    for i in range(0, len(list)):
        f = open("train-articles/" + list[i], encoding='utf8')
        texts.append(f.read())

    glove_6b50d = text.embedding.create("glove", pretrained_file_name='glove.6B.300d.txt')
    vectors = []
    for i in range(0, len(texts)):
        a = glove_6b50d.get_vecs_by_tokens(tokenizer.tokenize(texts[i])[0:1200])
        b = a.asnumpy()
        if (a.shape[0] < 1200):
            x = 1200 - int(a.shape[0])
            shape_zeros = np.zeros((x, 300))
            vector = np.vstack((b, shape_zeros))
            vector = vector.tolist()
        else:
            vector = b.tolist()
        vectors.append(vector)
    vectors = np.array(vectors)
    np.save("glove_300d_1200.npy", vectors)
    return vectors


def get_vector_test():
    # 文本
    texts = []
    list = os.listdir("dev-articles")
    for i in range(0, len(list)):
        f = open("dev-articles/" + list[i], encoding='utf8')
        texts.append(f.read())

    glove_6b50d = text.embedding.create("glove", pretrained_file_name='glove.6B.300d.txt')
    vectors = []
    for i in range(0, len(texts)):
        a = glove_6b50d.get_vecs_by_tokens(tokenizer.tokenize(texts[i])[0:1200])
        b = a.asnumpy()
        if (a.shape[0] < 1200):
            x = 1200 - int(a.shape[0])
            shape_zeros = np.zeros((x, 300))
            vector = np.vstack((b, shape_zeros))
            vector = vector.tolist()
        else:
            vector = b.tolist()
        vectors.append(vector)
    vectors = np.array(vectors)
    np.save("glove_test_300d_1200.npy", vectors)
    return vectors


def get_labels_vector():
    labels_vector_dict, texts = pre_deal_bert.get_labels_vector()
    labels_vector = []
    for key in labels_vector_dict.keys():
        labels_vector.append(labels_vector_dict[key])
    labels_vector = sequence.pad_sequences(np.array(labels_vector), maxlen=1200, padding='post')
    np.save("train_labels_vector_1200.npy", labels_vector)


def get_labels_vector_new():
    labels_vector_dict = new_pre_deal.get_labels_vector_new()
    labels_vector = []
    for key in labels_vector_dict.keys():
        labels_vector.append(labels_vector_dict[key])
    labels_vector = sequence.pad_sequences(np.array(labels_vector), maxlen=1200, padding='post')
    np.save("new_train_labels_vector_1200.npy", labels_vector)

def get_bert_labels_vector_new():
    labels_vector_dict = new_pre_deal.get_labels_vector_new()
    labels_vector = []
    for key in labels_vector_dict.keys():
        labels_vector.append(labels_vector_dict[key])
    labels_vector = sequence.pad_sequences(np.array(labels_vector), maxlen=500, padding='post')
    np.save("new_train_dev_labels_vector_500.npy", labels_vector)


def new_get_train_dev_vector():
    # 文本
    texts = []
    list = os.listdir("train_dev_articles")
    for i in range(0, len(list)):
        f = open("train_dev_articles/" + list[i], encoding='utf8')
        texts.append(f.read())

    glove_6b50d = text.embedding.create("glove", pretrained_file_name='glove.6B.300d.txt')
    vectors = []
    for i in range(0, len(texts)):
        a = glove_6b50d.get_vecs_by_tokens(tokenizer.tokenize(texts[i])[0:1200])
        b = a.asnumpy()
        if (a.shape[0] < 1200):
            x = 1200 - int(a.shape[0])
            shape_zeros = np.zeros((x, 300))
            vector = np.vstack((b, shape_zeros))
            vector = vector.tolist()
        else:
            vector = b.tolist()
        vectors.append(vector)
    vectors = np.array(vectors)
    np.save("new_train_dev_glove_300d_1200.npy", vectors)
    return vectors

def new_get_train_dev_vector_bert():
    # 文本
    texts = []
    list = os.listdir("train_dev_articles")
    for i in range(0, len(list)):
        f = open("train_dev_articles/" + list[i], encoding='utf8')
        texts.append(f.read())
    bc = BertClient(ip='222.19.197.229', port=5555, port_out=5556, check_version=False)
    texts_vector = bc.encode(texts)
    np.save("train_dev_vector_bert_450.npy", texts_vector)

    return texts_vector

def new_get_dev_vector_bert():
    # 文本
    texts = []
    list = os.listdir("dev_articles")
    for i in range(0, len(list)):
        f = open("dev_articles/" + list[i], encoding='utf8')
        texts.append(f.read())
    bc = BertClient(ip='222.19.197.228', port=5555, port_out=5556, check_version=False)
    texts_vector = bc.encode(texts)
    np.save("dev_vector_bert_500.npy", texts_vector)

    return texts_vector


def get_vector_train_tc():
    # 文本
    train_articles = pd.read_excel("mapping_TC.xlsx")
    text_list_train = []
    labels_list_train = []
    for i in range(0, 6369):
        text_list_train.append(str(train_articles['Associated_Propaganda'][i]))
        labels_list_train.append(train_articles['Classification'][i])

    glove_6b50d = text.embedding.create("glove", pretrained_file_name='glove.6B.300d.txt')
    vectors = []
    for i in range(0, len(text_list_train)):
        a = glove_6b50d.get_vecs_by_tokens(tokenizer.tokenize(text_list_train[i])[0:100])
        b = a.asnumpy()
        if (a.shape[0] < 100):
            x = 100 - int(a.shape[0])
            shape_zeros = np.zeros((x, 300))
            vector = np.vstack((b, shape_zeros))
            vector = vector.tolist()
        else:
            vector = b.tolist()
        vectors.append(vector)
    vectors = np.array(vectors)
    np.save("tc_glove_train.npy", vectors)
    return vectors


def get_vector_test_tc():
    # 文本
    dev_articles = pd.read_excel("TC_dev_predict.xlsx")
    text_list_dev = []
    for i in range(0, 1063):
        text_list_dev.append(dev_articles['Associated_Propaganda'][i])

    glove_6b50d = text.embedding.create("glove", pretrained_file_name='glove.6B.300d.txt')
    vectors = []
    for i in range(0, len(text_list_dev)):
        a = glove_6b50d.get_vecs_by_tokens(tokenizer.tokenize(text_list_dev[i])[0:100])
        b = a.asnumpy()
        if (a.shape[0] < 100):
            x = 100 - int(a.shape[0])
            shape_zeros = np.zeros((x, 300))
            vector = np.vstack((b, shape_zeros))
            vector = vector.tolist()
        else:
            vector = b.tolist()
        vectors.append(vector)
    vectors = np.array(vectors)
    np.save("tc_glove_test.npy", vectors)
    return vectors


def get_vector_test_final():
    # 文本
    texts = []
    list = os.listdir("test-articles")
    for i in range(0, len(list)):
        f = open("test-articles/" + list[i], encoding='utf8')
        texts.append(f.read())

    glove_6b50d = text.embedding.create("glove", pretrained_file_name='glove.6B.300d.txt')
    vectors = []
    for i in range(0, len(texts)):
        a = glove_6b50d.get_vecs_by_tokens(tokenizer.tokenize(texts[i])[0:1200])
        b = a.asnumpy()
        if (a.shape[0] < 1200):
            x = 1200 - int(a.shape[0])
            shape_zeros = np.zeros((x, 300))
            vector = np.vstack((b, shape_zeros))
            vector = vector.tolist()
        else:
            vector = b.tolist()
        vectors.append(vector)
    vectors = np.array(vectors)
    np.save("glove_final_300d_1200.npy", vectors)
    return vectors
def get_test():
    texts_vector = np.load("train_case_512.npy")
    return texts_vector
if __name__ == '__main__':
    # get_labels_vector_new()
    #new_get_train_dev_vector()
    #new_get_train_dev_vector_bert()
    #get_bert_labels_vector_new()
    text_vector = get_test()