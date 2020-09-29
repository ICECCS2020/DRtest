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
from scipy.misc import imsave, imread
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_data.svhn import data_svhn
from nmutant_attack.attacks import SaliencyMapMethod
from nmutant_util.utils_tf import model_argmax
from nmutant_model.model_operation import model_load
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1
from nmutant_data.data import get_shape
import time

FLAGS = flags.FLAGS

def jsma(datasets,sample_path, model_name, target,
         store_path='../mt_result/integration/jsma/mnist'):
    """
    the Jacobian-based saliency map approach (JSMA)
    :param datasets
    :param sample: inputs to attack
    :param target: the class want to generate
    :param nb_classes: number of output classes
    :return:
    """
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name)

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    if 'mnist' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(28,28,1)]).astype('float32')
        sample = preprocess_image_1(sample)
    elif 'cifar10' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(32,32,3)]).astype('float32')
        sample = preprocess_image_1(sample)
    elif 'svhn' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(32,32,3)]).astype('float32')
        sample = preprocess_image_1(sample)

    input_shape, nb_classes = get_shape(datasets)

    current_class = model_argmax(sess, x, preds, sample, feed=feed_dict)

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    if target == current_class:
        return 'The target is equal to its original class'
    elif target >= nb_classes or target < 0:
        return 'The target is out of range'

    print('Start generating adv. example for target class %i' % target)
    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    # This call runs the Jacobian-based saliency map approach
    one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
    one_hot_target[0, target] = 1
    jsma_params['y_target'] = one_hot_target
    adv_x = jsma.generate_np(sample, **jsma_params)

    # Check if success was achieved
    new_class_label = model_argmax(sess, x, preds, adv_x, feed=feed_dict)  # Predicted class of the generated adversary
    res = int(new_class_label == target)

    # Close TF session
    sess.close()
    if res == 1:
        adv_img_deprocessed = deprocess_image_1(adv_x)
        i = sample_path.split('/')[-1].split('.')[-2]
        path = store_path + '/adv_' + str(time.time()*1000) + '_' + i + '_' + str(current_class) + '_' + str(new_class_label) + '_.png'
        imsave(path, adv_img_deprocessed)
        print('$$$adv_img{' + path + '}')

    print('$$$ori_img{' + sample_path + '}')

def main(argv=None):
    datasets = FLAGS.datasets
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
        sample = X_test[0:1]
        imsave(FLAGS.sample, deprocess_image_1(sample))
    elif 'cifar10' == datasets:
        preprocess_image = preprocess_image_1
        train_start = 0
        train_end = 50000
        test_start = 0
        test_end = 10000

        # Get CIFAR10 test data
        X_train, Y_train, X_test, Y_test = data_cifar10(train_start=train_start,
                                                                           train_end=train_end,
                                                                           test_start=test_start,
                                                                           test_end=test_end,
                                                                           preprocess=preprocess_image)
        sample = X_test[198:199]
        imsave(FLAGS.sample, deprocess_image_1(sample))
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
        sample = X_test[198:199]
        imsave(FLAGS.sample, deprocess_image_1(sample))
    jsma(datasets = FLAGS.datasets,
         sample_path=FLAGS.sample,
         model_name=FLAGS.model,
         target=FLAGS.target,
         store_path=FLAGS.store_path)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('sample', '../datasets/integration/mnist/2.png', 'The path to load sample.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model.')
    flags.DEFINE_integer('target', 1, 'target')
    flags.DEFINE_string('store_path', '../mt_result/integration/jsma/mnist', 'The path to store adversaries.')

    tf.app.run()
