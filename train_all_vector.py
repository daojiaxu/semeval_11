import pre_deal
from nltk import tokenize
import KMP
from bert_serving.client import BertClient
import numpy as np

li = []
bc = BertClient(ip='222.19.197.230', port=5555, port_out=5556, check_version=False)
labels_vector_dict,test_text = pre_deal.get_labels_vector()
zero_vector = np.zeros((500, 768))
for i in range(0, len(test_text)):
    x = tokenize.word_tokenize(test_text[i])
    if (len(x) >502):
        index = KMP.KMP_algorithm(test_text[i], x[500] + " " + x[501])
        if (index != -1):
            list = []
            sentence_1 = test_text[i][0:index]
            sentence_2 = test_text[i][index:]
            list.append(sentence_1)
            list.append(sentence_2)
            vector = bc.encode(list)
            ve = np.concatenate((vector[0], vector[1]), axis=0)
            li.append(ve.tolist())
        else:
            list = []
            list.append(test_text[i])
            vector = bc.encode(list)
            ve = np.concatenate((vector[0], zero_vector), axis=0)
            li.append(ve.tolist())
    else:
        list = []
        list.append(test_text[i])
        vector = bc.encode(list)
        ve = np.concatenate((vector[0], zero_vector), axis=0)
        li.append(ve.tolist())
li_vector = np.array(li)
np.save("train_case_1000.npy", li_vector)
# x = tokenize.word_tokenize(test_text[2])
# index = KMP.KMP_algorithm(test_text[2], x[500] + " " + x[501])
# if (index != -1):
#     list = []
#     sentence_1 = test_text[2][0:index]
#     sentence_2 = test_text[2][index:]
#     list.append(sentence_1)
#     list.append(sentence_2)
#     vector = bc.encode(list)
#     ve = np.concatenate((vector[0], vector[0]), axis=0)
#     li.append(ve.tolist())
# else:
#     vector = bc.encode(test_text[2])
#     ve = np.concatenate((vector[0], zero_vector), axis=0)
#     li.append(ve.tolist())
