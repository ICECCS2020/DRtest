"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
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

from nmutant_attack.attacks import CarliniWagnerL2
from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_data.svhn import data_svhn
from nmutant_util.utils_tf import model_argmax
from nmutant_model.model_operation import model_load
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1
from nmutant_data.data import get_shape
import time

FLAGS = flags.FLAGS


def cw(datasets, sample, model_name, target,
       store_path='../mt_result/integration/cw/mnist'):
    """
    Carlini and Wagner's attack
    :param datasets
    :param sample: inputs to attack
    :param target: the class want to generate
    :param nb_classes: number of output classes
    :return:
    """
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name)

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
	'''
    if 'mnist' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(28, 28, 1)]).astype('float32')
        sample = preprocess_image_1(sample)
    elif 'cifar10' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(32, 32, 3)]).astype('float32')
        sample = preprocess_image_1(sample)
    elif 'svhn' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(32,32,3)]).astype('float32')
        sample = preprocess_image_1(sample)
    '''
    input_shape, nb_classes = get_shape(datasets)

    current_class = model_argmax(sess, x, preds, sample, feed=feed_dict)
    
    #if not os.path.exists(store_path):
    #    os.makedirs(store_path)

    if target == current_class:
        return 'The target is equal to its original class'
    elif target >= nb_classes or target < 0:
        return 'The target is out of range'
    print('succ@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('Start generating adv. example for target class %i' % target)
    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, back='tf', sess=sess)

    one_hot = np.zeros((1, nb_classes), dtype=np.float32)
    one_hot[0, target] = 1

    adv_inputs = sample
    adv_ys = one_hot
    yname = "y_target"
    if 'mnist' == datasets:
        cw_params = {'binary_search_steps': 1,
                     yname: adv_ys,
                     'max_iterations': 1000,
                     'learning_rate': 0.1,
                     'batch_size': 1,
                     'initial_const': 10}
    elif 'cifar10' == datasets:
        cw_params = {'binary_search_steps': 1,
                     yname: adv_ys,
                     'max_iterations': 1000,
                     'learning_rate': 0.1,
                     'batch_size': 1,
                     'initial_const': 0.1}
    print(adv_inputs.shape)
    adv = cw.generate_np(adv_inputs, **cw_params)
    print(adv_inputs.shape)
    print(adv.shape)
    new_class_labels = model_argmax(sess, x, preds, adv, feed=feed_dict)
    res = int(new_class_labels == target)

    # Close TF session
    sess.close()
    '''
    if res == 1:
        adv_img_deprocessed = deprocess_image_1(adv)
        i = sample_path.split('/')[-1].split('.')[-2]
        path = store_path + '/adv_' + str(time.time()*1000) + '_' + i + '_' + str(current_class) + '_' + str(new_class_labels) + '_.png'
        imsave(path, adv_img_deprocessed)
        print('$$$adv_img{' + path + '}')

    print('$$$ori_img{' + sample_path + '}')
    '''
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
        sample = X_test[0:10]
        #imsave(FLAGS.sample, deprocess_image_1(sample))
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
    cw(datasets=FLAGS.datasets,
       sample=FLAGS.sample,
       model_name=FLAGS.model,
       target=FLAGS.target,
       store_path=FLAGS.store_path)


if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('sample', '../datasets/integration/mnist/2.png', 'The path to load sample.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model.')
    flags.DEFINE_integer('target', 2, 'target')
    flags.DEFINE_string('store_path', '../mt_result/integration/cw/mnist', 'The path to store adversaries.')

tf.app.run()