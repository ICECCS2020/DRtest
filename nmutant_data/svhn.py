from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.io as sio

def y_one_hot(label):
    y = np.zeros(10)
    y[label] = 1
    return y

def data_svhn(datadir='../datasets/svhn/', train_start=0, train_end=73257, test_start=0,
                 test_end=26032, preprocess = None):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    preprocess_image = preprocess

    # train dataset
    train = sio.loadmat('../datasets/svhn/svhn_train.mat')
    X_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    X_train = np.asarray(preprocess_image(X_train.astype('float64')))
    Y_train = np.reshape(train['y'], (-1,)) - 1
    Y_train = np.asarray([y_one_hot(label) for label in Y_train])

    # test dataset
    test = sio.loadmat('../datasets/svhn/svhn_test.mat')
    X_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
    X_test = np.asarray(preprocess_image(X_test.astype('float64')))
    Y_test = np.reshape(test['y'], (-1,)) - 1
    Y_test = np.asarray([y_one_hot(label) for label in Y_test])

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    return X_train, Y_train, X_test, Y_test