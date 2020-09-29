from __future__ import absolute_import
from __future__ import print_function

import os
import sys
sys.path.append("../")
import tensorflow as tf
from tensorflow.python.platform import flags

import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nmutant_detection.das_util import block_split, train_lr, compute_roc_lid

FLAGS = flags.FLAGS

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar10': 0.26, 'svhn': 1.00}

def load_characteristics(dataset, attack):
    X, Y = None, None

    file_name = os.path.join('../detection/de/', "%s_%s_%.4f.npy" % (dataset, attack, BANDWIDTHS[dataset]))
    data = np.load(file_name)
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1]  # labels only need to load once

    return X, Y

def de_detect(datasets, model_path, sample_path, store_path, attack_type):

    print("Loading train attack: %s" % attack_type)

    X, Y = load_characteristics(datasets, attack_type)

    # standarization
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    # X = scale(X) # Z-norm

    # test attack is the same as training attack
    X_train, Y_train, X_test, Y_test = block_split(X, Y, int(len(X) * 0.4))

    print("Train data size: ", X_train.shape)
    print("Test data size: ", X_test.shape)

    ## Build detector
    print("LR Detector on [dataset: %s, attack: %s] with:" %(datasets, attack_type))
    lr = train_lr(X_train, Y_train)

    ## Evaluate detector
    y_pred = lr.predict_proba(X_test)[:, 1]
    y_label_pred = lr.predict(X_test)
    
    # AUC
    _, _, auc_score = compute_roc_lid(Y_test, y_pred, plot=False)
    precision = precision_score(Y_test, y_label_pred)
    recall = recall_score(Y_test, y_label_pred)

    y_label_pred = lr.predict(X_test)
    acc = accuracy_score(Y_test, y_label_pred)
    print('Detector ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (auc_score, acc, precision, recall))

    return lr, auc_score, scaler

def main(args):
    de_detect(FLAGS.datasets, FLAGS.model_path, FLAGS.sample_path, FLAGS.store_path, FLAGS.attack_type)

if __name__ == "__main__":
    # flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    # flags.DEFINE_string('model_path', '../models/integration/mnist', 'The path to load model.')
    # flags.DEFINE_string('sample_path', '../mt_result/integration/jsma/mnist', 'The path storing samples.')
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('model_path', '../models/integration/cifar10', 'The path to load model.')
    flags.DEFINE_string('sample_path', '../mt_result/integration/jsma/cifar10', 'The path storing samples.')
    flags.DEFINE_string('store_path', '../detection/', 'The path to store result.')
    flags.DEFINE_string('attack_type', 'jsma', 'attack_type')

    tf.app.run()
