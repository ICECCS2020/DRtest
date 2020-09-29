from __future__ import division, absolute_import, print_function

import sys

import tensorflow as tf
from tensorflow.python.platform import flags
sys.path.append("../")

import numpy as np
from nmutant_model.model_operation import model_load
from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1
from nmutant_detection import utils
from nmutant_util.utils_tf import model_argmax
from scipy.misc import imsave

FLAGS = flags.FLAGS

# Optimal KDE bandwidths that were determined from CV tuning
BANDWIDTHS = {'mnist': 1.20, 'cifar10': 0.26, 'svhn': 1.00}

def prepare_datasets(datasets, model_path, attack_type, sample_path):
    print('Loading the data and model...')
    # Load the model
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_path)
    # Load the dataset
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

    if attack_type == "normal":
        # Refine the normal, noisy and adversarial sets to only include samples for
        # which the original version was correctly classified by the model
        preds_test =np.asarray([])
        for i in range(40):
            preds_test = np.concatenate((preds_test, model_argmax(sess, x, preds, X_test[i*250:(i+1)*250], feed=feed_dict)))
        inds_correct = np.asarray(np.where(preds_test == Y_test.argmax(axis=1))[0])
        inds_correct = inds_correct[np.random.choice(len(inds_correct), 5000, replace=False)]
        X_test = X_test[inds_correct]
        for i in range(4000):
            imsave("../datasets/experiment/" + datasets + "/normal/train/" + str(inds_correct[i]) + '_' + str(int(preds_test[inds_correct[i]])) + '_' + str(int(preds_test[inds_correct[i]])) + '_.png', deprocess_image_1(X_test[i:i+1]))
        for j in range(1000):
            imsave("../datasets/experiment/" + datasets + "/normal/test/" + str(inds_correct[4000+j])+ '_' + str(int(preds_test[inds_correct[4000+j]]))+ '_' + str(int(preds_test[inds_correct[4000+j]])) + '_.png', deprocess_image_1(X_test[4000+j:4001+j]))
    elif attack_type == "error":
        preds_test = np.asarray([])
        for i in range(40):
            preds_test = np.concatenate(
                (preds_test, model_argmax(sess, x, preds, X_test[i * 250:(i + 1) * 250], feed=feed_dict)))
        inds_correct = np.asarray(np.where(preds_test != Y_test.argmax(axis=1))[0])
        X_test = X_test[inds_correct]
        num = int(len(X_test) * 0.8)
        for i in range(num):
            imsave("../datasets/experiment/" + datasets + "/error/train/" + str(inds_correct[i]) + '_' + str(
                int(np.argmax(Y_test[inds_correct[i]]))) + '_' + str(int(preds_test[inds_correct[i]])) + '_.png',
                   deprocess_image_1(X_test[i:i + 1]))
        for j in range(len(X_test) - num):
            imsave("../datasets/experiment/" + datasets + "/error/test/" + str(inds_correct[num + j]) + '_' + str(
                int(np.argmax(Y_test[inds_correct[num + j]]))) + '_' + str(int(preds_test[inds_correct[num + j]])) + '_.png',
                   deprocess_image_1(X_test[num + j:num+1 + j]))
    else:
        # Check attack type, select adversarial and noisy samples accordingly
        print('Loading adversarial samples...')
        # Load adversarial samplesx
        [X_test_adv, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(sample_path + attack_type + '/' + datasets)
        if len(X_test_adv) > 5000:
            index = np.asarray(range(len(X_test_adv)))
            index = index[np.random.choice(len(index), 5000, replace=False)]
            for i in range(4000):
                imsave("../datasets/experiment/" + datasets + "/" + attack_type + "/train/" + adv_image_files[index[i]], X_test_adv[index[i]])
            for j in range(1000):
                imsave("../datasets/experiment/" + datasets + "/" + attack_type + "/test/" + adv_image_files[index[4000+j]], X_test_adv[index[4000+j]])
        else:
            index = np.asarray(range(len(X_test_adv)))
            np.random.shuffle(index)
            cut = int(len(X_test_adv) * 0.8)
            for i in range(len(index)):
                if i < cut:
                    imsave("../datasets/experiment/" + datasets + "/" + attack_type + "/train/" + adv_image_files[index[i]], X_test_adv[index[i]])
                else:
                    imsave("../datasets/experiment/" + datasets + "/" + attack_type + "/test/" + adv_image_files[index[i]], X_test_adv[index[i]])

def main(args):
    prepare_datasets(FLAGS.datasets, FLAGS.model_path, FLAGS.attack_type, FLAGS.sample_path)

if __name__ == "__main__":
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('model_path', '../models/integration/cifar10', 'The path to load model.')
    flags.DEFINE_string('attack_type', 'jsma', 'attack_type')
    flags.DEFINE_string('sample_path', '../datasets/adversary/', 'The path to load samples')

    tf.app.run()

