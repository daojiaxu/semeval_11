from bert_serving.client import BertClient
import os
import numpy as np
import pre_deal
from keras.preprocessing import sequence

max_len =512
if __name__ == '__main__':
    labels_vector_dict, texts = pre_deal.get_labels_vector()
    labels_vector = []
    for key in labels_vector_dict.keys():
        labels_vector.append(labels_vector_dict[key])
    labels_vector = sequence.pad_sequences(np.array(labels_vector), maxlen=max_len, padding='post')
    # labels_vector = to_categorical(np.asarray(labels_vector), num_classes=2)
    bc = BertClient(ip='222.19.197.230', port=5555, port_out=5556, check_version=False)
    texts_vector = bc.encode(texts)
    np.save("case_vectors.npy",texts_vector)