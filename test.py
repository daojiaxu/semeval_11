import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import sys
import logging
import random

import numpy as np
import pandas as pd
import ast
import pickle
from keras import Sequential, layers
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

class_TC = []
f = open("propaganda-techniques-names-semeval2020task11.txt", encoding='utf8')
class_TC.append(f.read())
s = class_TC[0]
class_TC = s.split('\n')
class_TC.remove("")
list_labels = os.listdir("dev-articles")
filename = 'b.txt'
f = open(filename, 'w', encoding='utf-8')
for j in range(0, len(list_labels)):
    a = 10
    b = 100
    f.write(list_labels[j][7:16] + '\t' + class_TC[random.randint(0, 13)] + '\t' + str(a) + '\t' + str(
        b) + '\n')
f.close()
