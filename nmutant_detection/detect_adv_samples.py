from __future__ import division, absolute_import, print_function

import os
import sys

import tensorflow as tf
from tensorflow.python.platform import flags
sys.path.append("../")

import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nmutant_model.model_operation import model_load
from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_detection import utils
from nmutant_util.utils_tf import model_argmax

from nmutant_detection.das_util import (get_mc_predictions, get_deep_representations,
                                        score_samples, normalize, train_lr, compute_roc, merge_and_generate_labels, features, block_split)

FLAGS = flags.FLAGS

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar10': 0.26, 'svhn': 1.00}

def detect_adv_samples(datasets, model_path, sample_path, store_path, attack_type):
    print('Loading the data and model...')
    # Load the model
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_path)

    # # Load the dataset
    if 'mnist' == datasets:
        train_start = 0
        train_end = 60000
        test_start = 0
        test_end = 10000

        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                      train_end=train_end,
                                                      test_start=test_start,
                                                      test_end=test_end)
    elif 'cifar10' == datasets:
        preprocess_image = preprocess_image_1
        train_start = 0
        train_end = 50000
        test_start = 0
        test_end = 10000

        # Get CIFAR10 test data
        X_train, Y_train, fn_train, X_test, Y_test, fn_test = data_cifar10(train_start=train_start,
                                                                           train_end=train_end,
                                                                           test_start=test_start,
                                                                           test_end=test_end,
                                                                           preprocess=preprocess_image)

    # # Refine the normal, noisy and adversarial sets to only include samples for
    # # which the original version was correctly classified by the model
    # preds_test = model_argmax(sess, x, preds, X_test, feed=feed_dict)
    # inds_correct = np.where(preds_test == Y_test.argmax(axis=1))[0]
    # X_test = X_test[inds_correct]
    # X_test = X_test[np.random.choice(len(X_test), 500)]#500
    #
    # # Check attack type, select adversarial and noisy samples accordingly
    # print('Loading adversarial samples...')
    # # Load adversarial samplesx
    # [X_test_adv, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(sample_path)
    # X_test_adv = preprocess_image_1(np.asarray(X_test_adv).astype('float32'))
    # if len(X_test_adv.shape) < 4:
    #     X_test_adv = np.expand_dims(X_test_adv, axis=3)

    [X_test_adv_train, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test("../datasets/experiment/" + datasets + "/" + attack_type + "/train")
    [X_test_adv_test, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test("../datasets/experiment/" + datasets + "/" + attack_type + "/test")
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

    ## Get Bayesian uncertainty scores
    print('Getting Monte Carlo dropout variance predictions...')
    uncerts_normal = get_mc_predictions(sess, x, preds, X_test).var(axis=0).mean(axis=1)
    uncerts_adv = get_mc_predictions(sess, x, preds, X_test_adv).var(axis=0).mean(axis=1)

    ## Get KDE scores
    # Get deep feature representations
    print('Getting deep feature representations...')
    X_train_features = get_deep_representations(sess, x, X_train, model, feed_dict)
    X_test_normal_features = get_deep_representations(sess, x, X_test, model, feed_dict)
    X_test_adv_features = get_deep_representations(sess, x, X_test_adv, model, feed_dict)

    # Train one KDE per class
    print('Training KDEs...')
    class_inds = {}
    for i in range(Y_train.shape[1]):
        class_inds[i] = np.where(Y_train.argmax(axis=1) == i)[0]
    kdes = {}
    warnings.warn("Using pre-set kernel bandwidths that were determined "
                  "optimal for the specific CNN models of the paper. If you've "
                  "changed your model, you'll need to re-optimize the "
                  "bandwidth.")
    for i in range(Y_train.shape[1]):
        kdes[i] = KernelDensity(kernel='gaussian',
                                bandwidth=BANDWIDTHS[datasets]) \
            .fit(X_train_features[class_inds[i]])

    # Get model predictions
    print('Computing model predictions...')
    preds_test_normal = model_argmax(sess, x, preds, X_test, feed=feed_dict)
    preds_test_adv = model_argmax(sess, x, preds, X_test_adv, feed=feed_dict)

    # Get density estimates
    print('computing densities...')
    densities_normal = score_samples(
        kdes,
        X_test_normal_features,
        preds_test_normal
    )
    densities_adv = score_samples(
        kdes,
        X_test_adv_features,
        preds_test_adv
    )

    uncerts_pos = uncerts_adv[:]
    uncerts_neg = uncerts_normal[:]
    characteristics, labels = merge_and_generate_labels(uncerts_pos, uncerts_neg)
    file_name = os.path.join('../detection/bu/', "%s_%s.npy" % (datasets, attack_type))
    data = np.concatenate((characteristics, labels), axis=1)
    np.save(file_name, data)

    densities_pos = densities_adv[:]
    densities_neg = densities_normal[:]
    characteristics, labels = merge_and_generate_labels(densities_pos, densities_neg)
    file_name = os.path.join('../detection/de/', "%s_%s_%.4f.npy" % (datasets, attack_type, BANDWIDTHS[datasets]))
    data = np.concatenate((characteristics, labels), axis=1)
    np.save(file_name, data)

    ## Z-score the uncertainty and density values
    uncerts_normal_z, uncerts_adv_z = normalize(
        uncerts_normal,
        uncerts_adv
    )
    densities_normal_z, densities_adv_z = normalize(
        densities_normal,
        densities_adv
    )

    ## Build detector
    values, labels = features(
        densities_pos=densities_adv_z,
        densities_neg=densities_normal_z,
        uncerts_pos=uncerts_adv_z,
        uncerts_neg=uncerts_normal_z
    )
    X_tr, Y_tr, X_te, Y_te = block_split(values, labels, train_num)

    lr = train_lr(X_tr,Y_tr)

    ## Evaluate detector
    # Compute logistic regression model predictions
    probs = lr.predict_proba(X_te)[:, 1]
    preds = lr.predict(X_te)
    # Compute AUC
    n_samples = int(len(X_te) / 2)
    # The first 2/3 of 'probs' is the negative class (normal and noisy samples),
    # and the last 1/3 is the positive class (adversarial samples).
    _, _, auc_score = compute_roc(
        probs_neg=probs[:n_samples],
        probs_pos=probs[n_samples:]
    )

    precision = precision_score(Y_te, preds)
    recall = recall_score(Y_te, preds)

    y_label_pred = lr.predict(X_te)
    acc = accuracy_score(Y_te, y_label_pred)

    print('Detector ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (auc_score, acc, precision, recall))

def main(args):
    detect_adv_samples(FLAGS.datasets, FLAGS.model_path, FLAGS.sample_path, FLAGS.store_path, FLAGS.attack_type)

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

