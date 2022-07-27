import re
import numpy as np
import math

def tokenize(word_string):
    pattern = re.compile(r'[\w\d]+', re.I)
    return pattern.findall(word_string)

def make_tfidf_dict(c_list):

    assert type(c_list) == list, 'Функция принимает только итерируемый список слов'

    freq = {}
    tf = {}
    idf = {}
    len_c = len(c_list)
    len_docs = []
    
    for word_string in c_list:
        tokens = set(tokenize(word_string.strip().lower()))

        len_docs.append(len(tokens))

        for token in tokens:
            if token not in freq:
                freq[token] = 1
            else:
                freq[token] += 1


    to_sort = {i for i in freq.items()}
    freq_dict = sorted(to_sort, key = lambda x: (x[1], x[0]))

    num_feats = len(freq_dict)
    tf = np.zeros((len_c, num_feats))
    idf = np.zeros((len_c, num_feats))

    s = 0
    for word_string in c_list:
        tokens = tokenize(word_string.strip().lower())
        
        k = 0
        for i in freq_dict:
            tf[s, k] = tokens.count(i[0]) / len(tokens)
            idf[s, k] = num_feats / freq[i[0]]
            k += 1

        s += 1

    return tf, idf

def make_tfidf(tf, idf):

    tfidf = tf

    for l in range(tf.shape[0]):
        for c in range(tf.shape[1]):
            if tfidf[l, c] != 0:
                tfidf[l, c] = (np.log(tf[l, c]) + 1) * idf[l, c]

    return tfidf

def normalize_tfidf(tfidf):

    mean, stdev = np.mean(tfidf, axis = 1), np.std(tfidf, axis = 1)

    return ((tfidf.T - mean) / stdev).T

# Надо ли кол-во слов считать полностью? для TF

c_list = ['Казнить нельзя, помиловать. Нельзя наказывать.',
'Казнить, нельзя помиловать. Нельзя освободить.',
'Нельзя не помиловать.',
'Обязательно освободить.']

tf, idf = make_tfidf_dict(c_list)

tfidf = make_tfidf(tf, idf)

np.round(normalize_tfidf(tfidf), 2)