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
from nmutant_util.utils_tf import model_prediction, model_argmax, model_eval
from nmutant_model.model_operation import model_load
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1
from nmutant_data.data import get_shape, get_data
import time
import math

FLAGS = flags.FLAGS

def acc(datasets, model_name, target, attack):
    """
    Carlini and Wagner's attack
    :param datasets
    :param sample: inputs to attack
    :param target: the class want to generate
    :param nb_classes: number of output classes
    :return:
    """
    tf.reset_default_graph()
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    # sess, preds, x, y, model, feed_dict = model_load(datasets, model_name,de=True, epoch=target, attack=attack)
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, epoch=9)
    # sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, de=False, epoch=target, attack=attack)
    print(datasets)
    print('load successfule')
    eval_params = {'batch_size':256}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params, feed=feed_dict)

    sess.close()

    # tf.reset_default_graph()
    # X_train, Y_train, X_test, Y_test = get_data(datasets)
    # # sess, preds, x, y, model, feed_dict = model_load(datasets, model_name,de=True, epoch=target, attack=attack)
    # sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, epoch=64)
    # # sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, de=False, epoch=target, attack=attack)
    # print(datasets)
    # print('load successfule')
    # eval_params = {'batch_size': 256}
    #
    # preds_test = np.asarray([])
    # n_batches = int(np.ceil(1.0 * X_test.shape[0] / 256))
    # for i in range(n_batches):
    #     start = i * 256
    #     end = np.minimum(len(X_test), (i + 1) * 256)
    #     preds_test = np.concatenate(
    #         (preds_test, model_argmax(sess, x, preds, X_test[start:end], feed=feed_dict)))
    # inds_correct = np.asarray(np.where(preds_test == Y_test.argmax(axis=1))[0])
    print(accuracy)
    return accuracy#, len(inds_correct)
# def cw(datasets, sample, model_name, target,
#        store_path='../mt_result/integration/cw/mnist', ini_con=10, start=0, end=10000, batch_size=32):
#     """
#     Carlini and Wagner's attack
#     :param datasets
#     :param sample: inputs to attack
#     :param target: the class want to generate
#     :param nb_classes: number of output classes
#     :return:
#     """
#     tf.reset_default_graph()
#     X_train, Y_train, X_test, Y_test = get_data(datasets)
#     sess, preds, x, y, model, feed_dict = model_load(datasets, model_name,epoch=target)
#     print(datasets)
#     print('load successfule')
#     eval_params = {'batch_size':256}
#     accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params, feed=feed_dict)
#     print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
#     ###########################################################################
#     # Craft adversarial examples using Carlini and Wagner's approach
#     ###########################################################################
#
#     # input_shape, nb_classes = get_shape(datasets)
#     # sample=sample[start:end]
#
#     # sample = np.asarray([np.asarray(imread("/Users/pxzhang/Desktop/test/14_1.54598306318e+12_1_5.png")).reshape(28, 28, 1)]).astype('float32')
#     # sample = preprocess_image_1(sample)
#
#     #count = 0
#     #result = []
#     #for i in range(10000):
#      #   p = model_argmax(sess, x, preds, X_test[i:i + 1], feed=feed_dict)
#         #result.append(p)
#       #  if p == Y_test[i].argmax():
#        #     count = count + 1
#     #print(count)
#     # r = np.asarray(model_argmax(sess, x, preds, X_test[1000:2000], feed=feed_dict))
#     # c = 0
#     # for i in range (1000):
#     #     if r[i] == Y_test[i+ 1000].argmax():
#     #         c = c + 1
#     # print(c)
#     # result = np.asarray(result)
#     # print(r == result)
#
#
#     # print(model_argmax(sess, x, preds, X_test[0:1], feed=feed_dict))
#     # probabilities = model_prediction(sess, x, preds, X_test[0:1], feed=feed_dict)
#     # print(probabilities)
#     # print(np.argmax(probabilities, axis=1))
#     # path = '/Users/pxzhang/Documents/SUTD/project/nMutant/adv_result/mnist/cw/resnet50/8_0/test.jpg'
#     # adv_img_deprocessed = deprocess_image_1(X_test[0:1])
#     # adv_img_deprocessed = adv_img_deprocessed.reshape(adv_img_deprocessed.shape[1], adv_img_deprocessed.shape[2])
#     # imsave(path, adv_img_deprocessed)
#
#     # from nmutant_util.utils_file import get_data_file
#     # [image_list, image_files, real_labels, predicted_labels] = get_data_file("../adv_result/mnist/cw/" + model_name + "/10_0")
#     # samples_adv = np.asarray([image.astype('float64') for image in image_list])
#     # for i in range(len(image_list)):
#     #     print(model_argmax(sess, x, preds, samples_adv[i:i+1], feed=feed_dict), real_labels[i], predicted_labels[i])
#
#     # labels_adv = np.asarray([y_one_hot(int(label)) for label in real_labels])
#
#     sess.close()


def main(argv=None):
    print(acc('mnist', FLAGS.model, 0, 'fgsm'))
    # max = 0.0
    # max_index = 0
    # a={}
    # for i in range(FLAGS.end):
    #     accuracy = acc(datasets=FLAGS.datasets,
    #             model_name=FLAGS.model,
    #             target=str(i),
    #             attack=FLAGS.attack)
    #     if accuracy > 0.78 and i < 100:
    #         a[i] = accuracy
    #     if accuracy >= max:
    #         max_index = i
    #         max = accuracy
    # print(a)
    # print(max, max_index)



if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    # flags.DEFINE_string('sample', '../datasets/integration/mnist/0.png', 'The path to load sample.')
    flags.DEFINE_string('model', 'resnet152', 'The name of model.')
    flags.DEFINE_string('attack', 'fgsm', 'The name of attack.')
    flags.DEFINE_integer('end', 500, '')
    # flags.DEFINE_string('store_path', '../mt_result/integration/cw/mnist', 'The path to store adversaries.')

    tf.app.run()
