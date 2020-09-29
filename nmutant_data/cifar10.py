from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pickle as cPickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)#, encoding='bytes'
    return dict

def x_process(data):
    red = data[0:1024]
    green = data[1024:2048]
    blue = data[2048:3072]
    image = np.concatenate((red, green, blue)).reshape(3, 32, 32)
    image = np.transpose(image, (1, 2, 0))
    return image

def y_one_hot(label):
    y = np.zeros(10)
    y[label] = 1
    return y

def data_cifar10(datadir='../datasets/cifar10/', train_start=0, train_end=50000, test_start=0,
                 test_end=10000, preprocess = None):
    X_train = []
    Y_train = []
    fn_train = []
    X_test = []
    Y_test = []
    fn_test = []

    preprocess_image = preprocess

    # train dataset
    for i in range(5):
        dict = unpickle(datadir + 'data_batch_' + str(i+1))
        datas = dict.get(b'data')
        labels = dict.get(b'labels')
        fn_train = fn_train + dict.get(b'filenames')
        for j in range(len(datas)):
            X_train.append(preprocess_image(x_process(datas[j]).astype('float64')))
            Y_train.append(y_one_hot(labels[j]))

    # test dataset
    dict = unpickle(datadir + 'test_batch')
    datas = dict.get(b'data')
    labels = dict.get(b'labels')
    fn_test = fn_test + dict.get(b'filenames')
    for j in range(len(datas)):
        X_test.append(preprocess_image(x_process(datas[j]).astype('float64')))
        Y_test.append(y_one_hot(labels[j]))

    X_train = np.asarray(X_train[train_start:train_end])
    Y_train = np.asarray(Y_train[train_start:train_end])
    fn_train = np.asarray(fn_train[train_start:train_end])
    X_test = np.asarray(X_test[test_start:test_end])
    Y_test = np.asarray(Y_test[test_start:test_end])
    fn_test = np.asarray(fn_test[test_start:test_end])

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test