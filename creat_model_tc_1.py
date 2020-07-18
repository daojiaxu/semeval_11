import os
import sys
import logging
import random

import numpy as np
import pandas as pd
import ast
import pickle
from keras.layers.core import Dense, Activation
import tensorflow as tf
from bert_serving.client import BertClient
from keras.preprocessing import sequence

from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras_contrib.layers import CRF

from keras.layers import Embedding, Bidirectional, LSTM, Input, Masking, Dropout
from keras.models import Model
from keras.utils import to_categorical

from keras import optimizers

# from semeval.datasets import pre_deal
import pre_deal
import pre_deal_bert

_EPSILON = 1e-7
hidden_dim = 256
batch_size = 8
# batch_size = 32
nb_epoch = 15
max_len = 512


def create_model(maxlen, embedding_dim):
    sequence = Input(shape=(maxlen, embedding_dim,), dtype='float32')
    embedded = Masking(mask_value=0.)(sequence)

    hidden1 = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True, recurrent_dropout=0.25))(embedded)
    # hidden2 = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True, recurrent_dropout=0.25))(hidden1)

    Dense1 = Dropout(0.2)(hidden1)
    # Dense2 = Dense(64, activation='relu')(Dense1)
    # Dense3 = Dense(32, activation='relu')(Dense1)
    Dense4 = Dense(16, activation='relu')(Dense1)
    # crf = CRF(2, sparse_target=True, learn_mode='marginal')(Dense4)
    output = Dense(15, activation='softmax')(Dense4)
    # output=AMSoftmax(3,3,0.35)(hidden)
    model = Model(inputs=sequence, outputs=output)
    model.summary()

    return model


if __name__ == '__main__':
    # labels_vector_dict = pre_deal.get_train_vector_TC()
    labels_vector_dict = pre_deal_bert.get_train_vector_TC()
    labels_vector = []
    for key in labels_vector_dict.keys():
        labels_vector.append(labels_vector_dict[key])
    labels_vector = sequence.pad_sequences(np.array(labels_vector), maxlen=max_len, padding='post')
    # labels_vector = to_categorical(np.asarray(labels_vector), num_classes=2)
    # bc = BertClient(ip='222.19.197.230', port=5555, port_out=5556, check_version=False)
    # texts_vector = bc.encode(texts)
    # np.save("case_vectors.npy")
    texts_vector = np.load("train_case_512.npy")
    labels_vector = labels_vector.reshape(371, max_len, 1)
    labels_vector = to_categorical(labels_vector)
    # keras model
    model = create_model(max_len, 768)
    # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile("adam", loss="categorical_crossentropy", metrics=['accuracy'])
    model.fit(texts_vector, labels_vector, epochs=nb_epoch, batch_size=batch_size,
              validation_split=0.15)
    model.save('softmax_mode_3.h5')
    # test_textTokenWord = pre_deal.get_test_textVector()
    # test_vectors = bc.encode(test_textTokenWord)
    # print(model.predict(test_vectors))
    # print("--------------show---------------")
    # test_pred = model.predict(test_vectors).argmax(-1)
    # print("test_pred")
    # print(test_pred)
    # print(test_pred.shape)
