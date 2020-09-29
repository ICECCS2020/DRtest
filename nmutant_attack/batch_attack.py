"""
This tutorial shows how to generate adversarial examples
using JSMA in white-box setting.
The original paper can be found at:
https://arxiv.org/abs/1511.07528
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np
import tensorflow as tf
from scipy import ndimage
from scipy.misc import imsave
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_data.svhn import data_svhn
from nmutant_util.utils_tf import model_argmax
from nmutant_model.model_operation import model_load
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1
from nmutant_attack.blackbox import blackbox
from nmutant_attack.jsma import jsma
from nmutant_attack.cw import cw
from nmutant_attack.fgsm import fgsm

FLAGS = flags.FLAGS

def batch_attack(datasets, attack, model_path, store_path, nb_classes):
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
    elif 'svhn' == datasets:
        # choose the method of preprocess image
        preprocess_image = preprocess_image_1

        train_start = 0
        train_end = 73257
        test_start = 0
        test_end = 26032

        # Get SVHN test data
        X_train, Y_train, X_test, Y_test = data_svhn(train_start=train_start,
                                                     train_end=train_end,
                                                     test_start=test_start,
                                                     test_end=test_end,
                                                     preprocess=preprocess_image)

    store_path = store_path + attack + '/' + datasets
    sample_path = '../datasets/integration/batch_attack/' + datasets +'/'
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_path)
    if os.listdir(sample_path) == []:
        for i in range(len(X_test)):
            sample = X_test[i:i + 1]
            path = sample_path + str(i) + '.png'
            imsave(path, deprocess_image_1(sample))
            current_img = ndimage.imread(path)
            img = np.expand_dims(preprocess_image_1(current_img.astype('float64')),0)
            p = model_argmax(sess, x, preds, img, feed=feed_dict)
            if p != Y_test[i].argmax(axis=0):
                os.remove(path)
        # for i in range(len(X_test)):
        #     sample = X_test[i:i+1]
        #     if model_argmax(sess, x, preds, sample, feed=feed_dict) == Y_test[i].argmax(axis=0):
        #         path = sample_path + str(i) + '.png'
        #         imsave(path, deprocess_image_1(sample))

    sess.close()
    samples = os.listdir(sample_path)
    for sample in samples:
        tf.reset_default_graph()
        if 'blackbox' == attack:
            blackbox(datasets=datasets,
                     sample_path=sample_path + sample,
                     model_path=model_path,
                     store_path=store_path,
                     nb_classes=nb_classes)
        elif 'fgsm' == attack:
            fgsm(datasets=datasets,
                 sample_path=sample_path + sample,
                 model_path=model_path,
                 store_path=store_path,
                 nb_classes=nb_classes)
        else:
            i = int(sample.split('.')[-2])
            for j in range(nb_classes):
                tf.reset_default_graph()
                if Y_test[i][j] == 0:
                    if 'jsma' == attack:
                        jsma(datasets = datasets,
                             sample_path=sample_path + sample,
                             target = j,
                             model_path=model_path,
                             store_path=store_path,
                             nb_classes = nb_classes)
                    if 'cw' == attack:
                        cw(datasets=datasets,
                           sample_path=sample_path + sample,
                           target=j,
                           model_path=model_path,
                           store_path=store_path,
                           nb_classes=nb_classes)

def main(argv=None):
    batch_attack(datasets=FLAGS.datasets, attack=FLAGS.attack, model_path=FLAGS.model_path, store_path=FLAGS.store_path, nb_classes=FLAGS.nb_classes)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('attack', 'jsma', 'The type of generating adversaries')
    flags.DEFINE_string('model_path', '../models/integration/mnist', 'The path to load model.')
    flags.DEFINE_string('store_path', '../mt_result/integration/', 'The path to store adversaries.')
    flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')

    tf.app.run()
