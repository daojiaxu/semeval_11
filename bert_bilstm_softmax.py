# bert-serving-start -pooling_strategy NONE -model_dir E:\lib\bert_models\uncased_L-12_H-768_A-12

import os
import sys
import logging

import numpy as np
import pandas as pd
import ast
import pickle

from utils import load_train_file, load_test_file
from bert_serving.client import BertClient
from keras.preprocessing import sequence

from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras_contrib.layers import CRF

from keras.layers import Embedding, Bidirectional, LSTM, Input, Masking, Dense
from keras.models import Model
from keras.utils import np_utils

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from bert_serving.client import BertClient

bc = BertClient(ip='222.19.197.230', port=5555, port_out=5556,check_version=False)
hidden_dim = 256
batch_size = 32
# nb_epoch = 1
nb_epoch = 50


def create_model(maxlen, embedding_dim):
    sequence = Input(shape=(maxlen, embedding_dim,), dtype='float32')
    embedded = Masking(mask_value=0.)(sequence)

    hidden = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True, recurrent_dropout=0.25)) (embedded)
    hidden = Bidirectional(LSTM(hidden_dim // 2, return_sequences=True, recurrent_dropout=0.25)) (hidden)

    # crf = CRF(len(label_tag_dict), sparse_target=True) (hidden)
    output = Dense(len(label_tag_dict), activation='softmax') (hidden)

    model = Model(inputs=sequence, outputs=output)
    return model


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'twitter_bert.pickle3')
    train_vectors, test_vectors, train_labels, test_labels, label_tag_dict, tag_label_dict = \
            pickle.load(open(pickle_file, 'rb'))

    train_labels = np_utils.to_categorical(np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1])))
    print(train_labels.shape)

    maxlen = train_vectors.shape[1]
    embedding_dim = train_vectors.shape[2]
    val_test_labels = sequence.pad_sequences(np.array(test_labels), maxlen=maxlen, padding='post')
    val_test_labels = np_utils.to_categorical(val_test_labels)
    # val_test_labels = np.reshape(val_test_labels, (val_test_labels.shape[0], val_test_labels.shape[1], 1))

    # keras model
    model = create_model(maxlen, embedding_dim)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_vectors, train_labels, epochs=nb_epoch, batch_size=batch_size, validation_data=[test_vectors, val_test_labels])

    test_pred = model.predict(test_vectors).argmax(-1)
    print(test_pred)

    y_true, y_pred = [], []
    for i, labels in enumerate(test_labels):
        for j, label in enumerate(labels):
            y_pred.append(test_pred[i][j])
            y_true.append(label)

    print(np.unique(y_pred))

    logging.info('classes f1_score: ' + str(f1_score(y_true, y_pred, average=None)))
    logging.info('classes precision_score: ' + str(precision_score(y_true, y_pred, average=None)))
    logging.info('classes recall_score: ' + str(recall_score(y_true, y_pred, average=None)))

    logging.info('f1_score: ' + str(f1_score(y_true, y_pred, average='micro')))
    logging.info('precision_score: ' + str(precision_score(y_true, y_pred, average='micro')))
    logging.info('recall_score: ' + str(recall_score(y_true, y_pred, average='micro')))

    logging.info('f1_score: ' + str(f1_score(y_true, y_pred, average='weighted')))
    logging.info('precision_score: ' + str(precision_score(y_true, y_pred, average='weighted')))
    logging.info('recall_score: ' + str(recall_score(y_true, y_pred, average='weighted')))

    logging.info('accuracy_score: ' + str(accuracy_score(y_true, y_pred)))

    logging.info(precision_recall_fscore_support(y_true, y_pred, beta=1, average='micro'))
    print(classification_report(y_true, y_pred))