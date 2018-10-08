#!/usr/bin/python
# coding=utf-8
# @Time : 2018/3/8 下午3:02
# @Author : ComeOnJian
# @File : process_data.py

import pickle
# import word2vec
import numpy as np
from collections import defaultdict,OrderedDict
import re
from tqdm import tqdm
import pandas as pd


data_dir = '../data/rt-polaritydata/'
google_new_vector_dir = '../data/'

##方式一，编写代码加载word2vec训练好的模型文件GoogleNews-vectors-negative300.bin
####参照word2vec.from_binary方法改写
###谷歌预训练好的词向量数据 GoogleNews-vectors-negative300.bin.gz 。该词向量是基于谷歌新闻数据用word2vec(cbow方法：由语境推测目标词汇，最后学到词的embedding空间)训练所得：
#http://ju.outofmemory.cn/entry/321125
def load_binary_vec(fname, vocab):#fname为GoogleNews-vectors-negative300.bin.gz   vocab为训练集训练后处理得到的词汇表
    word_vecs = {}
    with open(fname, 'rb') as fin:
        header = fin.readline()
        vocab_size, vector_size = list(map(int, header.split()))
        binary_len = np.dtype(np.float32).itemsize * vector_size#将每个词占用的二进制长度记录下来
        # vectors = []
        for i in tqdm(range(vocab_size)):#这是什么？词的个数
            # read word
            word = b''
            while True:
                ch = fin.read(1)#？
                if ch == b' ':
                    break
                word += ch
            print(">>>>>"+str(word))
            word = word.decode(encoding='ISO-8859-1')#若google_news向量化表示后的某个词存在于经过训练集训练后处理得到的词汇表vocab中，则记录下来，存到word_vecs字典中
            if word in vocab:
                word_vecs[word] = np.fromstring(fin.read(binary_len), dtype=np.float32)#读到了目标词在embedding中的向量化表示
            else:
                fin.read(binary_len)#过滤掉改词的向量化表示
            # vector = np.fromstring(fin.read(binary_len), dtype=np.float32)
            # vectors.append(vector)
            # if include:
            #     vectors[i] = unitvec(vector)
            fin.read(1)  # newline
        return word_vecs#

#load MR data —— Movie reviews with one sentence per review.
#Classification involves detecting positive/negative reviews
def load_data_k_cv(folder,cv=10,clear_flag=True):
    pos_file = folder[0]
    neg_file = folder[1]

    #训练集的语料词汇表,计数
    word_cab=defaultdict(float)#定义词汇字典
    revs = [] #最后的数据，元素类型为json格式，代表每个句子的一些基本属性
    with open(pos_file,'rb') as pos_f:
        for line in pos_f:
            rev = []
            rev.append(line.decode(encoding='ISO-8859-1').strip())#读取文本内一行数据
            if clear_flag:#为1表示已经预处理过
                orign_rev = clean_string(" ".join(rev))#进行正则匹配返回
            else:
                orign_rev = " ".join(rev).lower()
            words = set(orign_rev.split())#分为单字
            for word in words:
                word_cab[word] += 1
            datum = {"y": 1,#positive数据标识
                     "text": orign_rev,
                     "num_words": len(orign_rev.split()),#句子的长度
                     "spilt": np.random.randint(0, cv)}
            revs.append(datum)

    with open(neg_file,'rb') as neg_f:
        for line in neg_f:
            rev = []
            rev.append(line.decode(encoding='ISO-8859-1').strip())
            if clear_flag:
                orign_rev = clean_string(" ".join(rev))
            else:
                orign_rev = " ".join(rev).lower()
            words = set(orign_rev.split())
            for word in words:
                word_cab[word] += 1
            datum = {"y": 0,
                     "text": orign_rev,
                     "num_words": len(orign_rev.split()),
                     "spilt": np.random.randint(0, cv)}#返回0-10之间的随机数
            revs.append(datum)

    return word_cab,revs

def add_unexist_word_vec(w2v,vocab):
    """
    将词汇表中没有embedding的词初始化()
    :param w2v:经过word2vec训练好的词向量
    :param vocab:总体要embedding的词汇表
    """
    for word in vocab:#vocab中的元素形式为{"china":3,"day":0,........}
        if word not in w2v and vocab[word]>=1:#w2v中的元素形式为 {"china":[0,3,4,3..];"day":[2,4,5,....]}
            w2v[word] = np.random.uniform(-0.25,0.25,300)#300维，按照均匀分布随机初始化

def clean_string(string,TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def get_vec_by_sentence_list(word_vecs,sentence_list,maxlen=56,values=0.,vec_size = 300):
    """
    :param sentence_list:句子列表
    :return:句子对应的矩阵向量表示
    """
    data = []

    for sentence in sentence_list:
        # get a sentence
        sentence_vec = []
        words = sentence.split()
        for word in words:
            sentence_vec.append(word_vecs[word].tolist())#将词向量添加进入句子的表示矩阵上

        # padding sentence vector to maxlen(w * h)
        sentence_vec = pad_sentences(sentence_vec,maxlen,values,vec_size)
        # add a sentence vector
        data.append(np.array(sentence_vec))
    return data

def get_index_by_sentence_list(word_ids,sentence_list,maxlen=56):
    indexs = []
    words_length = len(word_ids)
    for sentence in sentence_list:
        # get a sentence
        sentence_indexs = []
        words = sentence.split()
        for word in words:
            sentence_indexs.append(word_ids[word])
        # padding sentence to maxlen
        length = len(sentence_indexs)
        if length < maxlen:
            for i in range(maxlen - length):
                sentence_indexs.append(words_length)#直接加上一个数？？可能会自动转化吧
        # add a sentence vector
        indexs.append(sentence_indexs)

    return np.array(indexs)

    pass
def pad_sentences(data,maxlen=56,values=0.,vec_size = 300):
    """padding to max length将一个句子补充到最长数目56个词
    :param data:要扩展的数据集
    :param maxlen:扩展的h长度
    :param values:默认的值
    """
    length = len(data)
    if length < maxlen:
        for i in range(maxlen - length):
            data.append(np.array([values]*vec_size))#添加向量[0,0,0,0,....0]
    return data

def get_train_test_data1(word_vecs,revs,cv_id=0,sent_length = 56,default_values=0.,vec_size = 300):
    """
    获取的训练数据和测试数据是直接的数据
    :param revs:
    :param cv_id:
    :param sent_length:
    :return:
    """
    data_set_df = pd.DataFrame(revs)

    # DataFrame
    # y text num_words spilt
    # 1 'I like this movie' 4 3
    data_set_df = data_set_df.sample(frac=1)#打乱顺序

    data_set_cv_train = data_set_df[data_set_df['spilt'] != cv_id]  # 训练集
    data_set_cv_test = data_set_df[data_set_df['spilt'] == cv_id] #测试集

    # train
    train_y_1 = np.array(data_set_cv_train['y'].tolist(),dtype='int')
    train_y_2 = list(map(get_contrast,train_y_1))
    train_y = np.array([train_y_1,train_y_2]).T

    test_y_1 = np.array(data_set_cv_test['y'].tolist(),dtype='int')
    test_y_2 = list(map(get_contrast,test_y_1))
    test_y = np.array([test_y_1,test_y_2]).T

    train_sentence_list = data_set_cv_train['text'].tolist()
    test_sentence_list = data_set_cv_test['text'].tolist()


    train_x = get_vec_by_sentence_list(word_vecs,train_sentence_list,sent_length,default_values,vec_size)
    test_x = get_vec_by_sentence_list(word_vecs,test_sentence_list,sent_length,default_values,vec_size)


    return train_x,train_y,test_x,test_y
def get_train_test_data2(word_ids,revs,cv_id=0,sent_length = 56):
    data_set_df = pd.DataFrame(revs)

    # DataFrame
    # y text num_words spilt
    # 1 'I like this movie' 4 3
    data_set_df = data_set_df.sample(frac=1)  # 打乱顺序

    data_set_cv_train = data_set_df[data_set_df['spilt'] != cv_id]  # 训练集
    data_set_cv_test = data_set_df[data_set_df['spilt'] == cv_id]  # 测试集

    # train
    train_y_1 = np.array(data_set_cv_train['y'].tolist(), dtype='int')
    train_y_2 = list(map(get_contrast, train_y_1))
    train_y = np.array([train_y_1, train_y_2]).T

    test_y_1 = np.array(data_set_cv_test['y'].tolist(), dtype='int')
    test_y_2 = list(map(get_contrast, test_y_1))
    test_y = np.array([test_y_1, test_y_2]).T

    train_sentence_list = data_set_cv_train['text'].tolist()
    test_sentence_list = data_set_cv_test['text'].tolist()

    train_x = get_index_by_sentence_list(word_ids,train_sentence_list,sent_length)
    test_x = get_index_by_sentence_list(word_ids,test_sentence_list,sent_length)

    return train_x,train_y,test_x,test_y
#对0，1取反
def get_contrast(x):
    return (2+~x)

def getWordsVect(W):
    word_ids = OrderedDict()
    W_list = []
    count =0
    for word,vector in W.items():
        W_list.append(vector.tolist())
        word_ids[word] = count
        count = count + 1
    W_list.append([0.0]*300)
    return word_ids,W_list




if __name__ == '__main__':

    #Testing --------------------------------------------------------------------------

#   ./表示当前目录
# ../表示父级目录
    data_folder = ['../data/rt-polaritydata/rt-polarity.pos', '../data/rt-polaritydata/rt-polarity.neg']
    w2v_file = '../data/GoogleNews-vectors-negative300.bin'
    print('load data ...')

    word_cab, revs = load_data_k_cv(data_folder)#训练集的word_cab#定义词汇字典# revs训练集的最后数据格式，元素类型为json格式，每个元素统计了每个句子的基本属性，如长度等
    print('data loaded !!!')
    print('number of sentences: ' + str(len(revs)))
    print('size of vocab: ' + str(len(word_cab)))
    sentence_max_len = np.max(pd.DataFrame(revs)['num_words'])#最大的句子长度
    print('dataset the sentence max length is {}'.format(sentence_max_len))

    print('load word2vec vectors...')
    word_vecs = load_binary_vec(w2v_file,word_cab)##返回goolenew通过word2vec训练后获得得词向量化表示
    print (len(list(word_vecs.keys())))
    print('finish word2vec load !!!')

    #对未登录词操作，随机初始化为出现在向量化表中的词
    add_unexist_word_vec(word_vecs,word_cab)

    # CNN-rand对应的词向量表
    word_vecs_rand = {}

    add_unexist_word_vec(word_vecs_rand,word_cab)#即word_vecs_rand中存的是词汇表word_cab中所有的词随机均匀向量化后得到的向量化表示，为词典形式{china:[2,2,2,3,4,..],japan:[2,4,5,6,7,....]....}

    #将数据数据集对应的词向量保存好
    pickle.dump([word_vecs_rand,word_vecs,word_cab,sentence_max_len,revs],open('../result/word_vec.p','wb'))




    pass