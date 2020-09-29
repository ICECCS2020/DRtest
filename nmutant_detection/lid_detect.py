from __future__ import absolute_import
from __future__ import print_function

import os
import sys

sys.path.append("../")
import tensorflow as tf
from tensorflow.python.platform import flags

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nmutant_detection.das_util import block_split, train_lr, compute_roc_lid, merge_and_generate_labels, get_lids_random_batch
from nmutant_model.model_operation import model_load
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_detection import utils

FLAGS = flags.FLAGS

def get_lid(sess, x, model, feed_dict, X_test, X_test_adv, k=20, batch_size=100, dataset='mnist'):
    """
    Get local intrinsic dimensionality
    :param model:
    :param X_train:
    :param Y_train:
    :param X_test:
    :param X_test_noisy:
    :param X_test_adv:
    :return: artifacts: positive and negative examples with lid values,
            labels: adversarial (label: 1) and normal/noisy (label: 0) examples
    """
    print('Extract local intrinsic dimensionality: k = %s' % k)
    lids_normal, lids_adv = get_lids_random_batch(sess, x, model, feed_dict, X_test, X_test_adv, dataset, k, batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_adv:", lids_adv.shape)

    artifacts, labels = merge_and_generate_labels(lids_adv, lids_normal)

    return artifacts, labels

def calculate_lid(datasets, model_path, sample_path, attack, k_nearest, batch_size):
    """
    Load multiple characteristics for one dataset and one attack.
    :param dataset: 
    :param attack: 
    :param characteristics: 
    :return: 
    """
    # Load the model
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_path)

    [X_test_adv_train, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test("../datasets/experiment/" + datasets + "/" + attack + "/train")
    [X_test_adv_test, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test("../datasets/experiment/" + datasets + "/" + attack+ "/test")
    train_num = len(X_test_adv_train)
    test_num = len(X_test_adv_test)
    X_test_adv = preprocess_image_1(np.concatenate((np.asarray(X_test_adv_train),np.asarray(X_test_adv_test))).astype('float32'))
    if len(X_test_adv.shape) < 4:
        X_test_adv = np.expand_dims(X_test_adv, axis=3)

    [X_test_train, adv_image_files, real_labels, predicted_labels] = utils.get_data_normal_test("../datasets/experiment/" + datasets + "/normal/train")
    [X_test_test, adv_image_files, real_labels, predicted_labels] = utils.get_data_normal_test("../datasets/experiment/" + datasets + "/normal/test")
    X_test_train = np.asarray(X_test_train)[np.random.choice(len(X_test_train), train_num, replace=False)]
    X_test_test = np.asarray(X_test_test)[np.random.choice(len(X_test_test), test_num, replace=False)]
    X_test = preprocess_image_1(np.concatenate((np.asarray(X_test_train),np.asarray(X_test_test))).astype('float32'))
    if len(X_test.shape) < 4:
        X_test = np.expand_dims(X_test, axis=3)

    file_name = os.path.join('../detection/lid/', "%s_%s.npy" % (datasets, attack))
    if not os.path.exists(file_name):
        # extract local intrinsic dimensionality
        characteristics, labels = get_lid(sess, x, model, feed_dict, X_test, X_test_adv, k_nearest, batch_size, datasets)
        data = np.concatenate((characteristics, labels), axis=1)
        np.save(file_name, data)
    return train_num

def load_characteristics(dataset, attack):
    X, Y = None, None

    file_name = os.path.join('../detection/lid/', "%s_%s.npy" % (dataset, attack))
    data = np.load(file_name)
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1]  # labels only need to load once

    return X, Y

def lid_detect(datasets, model_path, sample_path, store_path, attack_type, k_nearest, batch_size):

    print("Loading train attack: %s" % attack_type)
    num = calculate_lid(datasets, model_path, sample_path, attack_type, k_nearest, batch_size)

    X, Y = load_characteristics(datasets, attack_type)

    # standarization
    scaler = MinMaxScaler().fit(X)
    X = scaler.transform(X)
    # X = scale(X) # Z-norm

    # test attack is the same as training attack
    X_train, Y_train, X_test, Y_test = block_split(X, Y, num)

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
    lid_detect(FLAGS.datasets, FLAGS.model_path, FLAGS.sample_path, FLAGS.store_path, FLAGS.attack_type, FLAGS.k_nearest, FLAGS.batch_size)

if __name__ == "__main__":
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model_path', '../models/integration/mnist', 'The path to load model.')
    flags.DEFINE_string('sample_path', '../mt_result/integration/jsma/mnist', 'The path storing samples.')
    # flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    # flags.DEFINE_string('model_path', '../models/integration/cifar10', 'The path to load model.')
    # flags.DEFINE_string('sample_path', '../mt_result/integration/jsma/cifar10', 'The path storing samples.')
    flags.DEFINE_string('store_path', '../detection/', 'The path to store result.')
    flags.DEFINE_string('attack_type', 'jsma', 'attack_type')
    flags.DEFINE_integer('k_nearest', 20, 'The number of nearest neighbours to use.')
    flags.DEFINE_integer('batch_size', 100, 'The batch size to use for training detector.')

    tf.app.run()
