import os
from nltk import tokenize
import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

def get_labels_vector():
    # 文本
    texts = []
    list = os.listdir("train-articles")
    for i in range(0, len(list)):
        f = open("train-articles/" + list[i], encoding='utf8')
        texts.append(f.read())

    # 标签
    labels = []
    list_labels = os.listdir("train-labels-task1-span-identification")
    for i in range(0, len(list_labels)):
        f = open("train-labels-task1-span-identification/" + list_labels[i], encoding='utf8')
        labels.append(f.read())

    labels_tag = {}  # 存储每篇文章的分词
    for i in range(0, len(list_labels)):
        labels_tag[list_labels[i][7:16]] = []

    for i in range(0, len(texts)):
        labels_tag[list_labels[i][7:16]].append(tokenizer.tokenize(texts[i]))

    labels_tag_qujian = {}  # 存储每篇文章的重点区间
    for i in range(0, len(list_labels)):
        labels_tag_qujian[list_labels[i][7:16]] = []
    i = 10
    j = 10
    k = 0
    for x in range(0, len(list_labels)):
        if k < len(list_labels):
            try:
                while (True):

                    while (labels[k][i] != '\t' and labels[k][i] != '\n'):
                        if i < (len(labels[k]) - 1):
                            i = i + 1
                        else:
                            break
                    # print(j, i)
                    # print(labels[k][j:i])
                    if (len(labels[k][j:i]) < 6):
                        labels_tag_qujian[list_labels[k][7:16]].append(labels[k][j:i])
                    j = i + 1
                    i = j
            except IndexError:
                None
            i = 10
            j = 10
            k = k + 1

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
    return labels_tag_word_vector, texts



def get_test_textVector():
    # 文本
    texts = []
    list = os.listdir("dev-articles")
    for i in range(0, len(list)):
        f = open("dev-articles/" + list[i], encoding='utf8')
        texts.append(f.read())

    # texts_token = []
    # for i in range(0, len(texts)):
    #     texts_token.append(tokenizer.tokenize(texts[i]))

    return texts

def new_get_test_textVector():
    # 文本
    texts = []
    list = os.listdir("test-articles")
    for i in range(0, len(list)):
        f = open("test-articles/" + list[i], encoding='utf8')
        texts.append(f.read())

    # texts_token = []
    # for i in range(0, len(texts)):
    #     texts_token.append(tokenizer.tokenize(texts[i]))

    return texts

def get_train_vector_TC():
    train_articles = pd.read_excel("mapping_TC.xlsx")

    # In[25]:

    train_articles.head(8)

    # In[15]:

    # 文本
    texts = []
    list = os.listdir("train-articles")
    for i in range(0, len(list)):
        f = open("train-articles/" + list[i], encoding='utf8')
        texts.append(f.read())

    # In[16]:

    # 标签
    labels = []
    list_labels = os.listdir("train-labels-task2-technique-classification")
    for i in range(0, len(list_labels)):
        f = open("train-labels-task2-technique-classification/" + list_labels[i], encoding='utf8')
        labels.append(f.read())

    # In[18]:

    text_dict = {}  # 存储每篇文章的分词
    for i in range(0, len(list_labels)):
        text_dict[list_labels[i][7:16]] = texts[i]

    # In[33]:

    id_lst = []
    start_lst = []
    end_lst = []
    for i in range(0, train_articles.shape[0]):
        a = len(tokenizer.tokenize(
            text_dict[str(train_articles['File_ID'][i])][:int(train_articles['Start_IDX'][i])]))  # 起始区间
        b = len(tokenizer.tokenize(text_dict[str(train_articles['File_ID'][i])][
                                       int(train_articles['Start_IDX'][i]):int(train_articles['End_IDX'][i])]))
        c = a + b  # 终点区间
        start_lst.append(a)
        end_lst.append(c)
        id_lst.append(train_articles['File_ID'][i])

    # In[36]:

    frame_ = pd.DataFrame({'File_ID': id_lst, 'Start_IDX_Word': start_lst, 'End_IDX_Word': end_lst})
    writer = pd.ExcelWriter('mapping_TC_word.xlsx', engine='xlsxwriter')
    frame_.to_excel(writer, sheet_name='task-2_word')
    writer.save()

    # In[37]:

    labels_tag_word_vector = {}
    for i in range(0, len(list_labels)):
        labels_tag_word_vector[list_labels[i][7:16]] = []

    for j in range(0, len(list_labels)):
        for length in range(0, len(tokenizer.tokenize(texts[j]))):
            labels_tag_word_vector[list_labels[j][7:16]].append(0)

    # In[39]:

    train_articles_word = pd.read_excel("mapping_TC_word.xlsx", )

    # In[40]:

    train_articles_word.head()

    # In[70]:

    class_TC = []
    f = open("propaganda-techniques-names-semeval2020task11.txt", encoding='utf8')
    class_TC.append(f.read())
    s = class_TC[0]
    class_TC = s.split('\n')
    class_TC.remove("")

    # In[83]:

    for i in range(0, train_articles_word.shape[0]):
        a = len(tokenizer.tokenize(text_dict[str(train_articles_word['File_ID'][i])]))
        if (int(train_articles_word['Start_IDX_Word'][i]) < a and int(train_articles_word['End_IDX_Word'][i]) + 1 < a):
            for j in range(0, len(class_TC)):
                if (train_articles['Classification'][i] == class_TC[j]):
                    index = j + 1
            for length in range(int(train_articles_word['Start_IDX_Word'][i]),
                                int(train_articles_word['End_IDX_Word'][i]) + 1):
                labels_tag_word_vector[str(train_articles_word['File_ID'][i])][length] = index
    return labels_tag_word_vector