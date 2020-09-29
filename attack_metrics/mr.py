"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
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
from nmutant_util.utils_tf import model_argmax, model_prediction
from nmutant_model.model_operation import model_load
from nmutant_attack.attacks import FastGradientMethod
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1, deprocess_image_1
from nmutant_data.data import get_data, get_shape
from nmutant_util.utils import batch_indices
from nmutant_util.utils_file import get_data_file
import time
import math

FLAGS = flags.FLAGS
    
def mr(datasets, model_name, attack, va, epoch=49):
    """
    :param datasets
    :param sample: inputs to attack
    :param target: the class want to generate
    :param nb_classes: number of output classes
    :return:
    """
    tf.reset_default_graph()
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    input_shape, nb_classes = get_shape(datasets)
    sample=X_test
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, epoch=epoch)
    probabilities = model_prediction(sess, x, preds, sample, feed=feed_dict, datasets=datasets)

    if sample.shape[0] == 1:
        current_class = np.argmax(probabilities)
    else:
        current_class = np.argmax(probabilities, axis=1)
    # only for correct:
    acc_pre_index=[]
    for i in range(0, sample.shape[0]):
        if current_class[i]==np.argmax(Y_test[i]):
            acc_pre_index.append(i)
    print(len(acc_pre_index))
    sess.close()
    total=0

    if attack=='fgsm':
            
        samples_path='../adv_result/'+datasets+'/'+attack+'/'+model_name+'/'+str(va)
        [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
        num=len(image_list)
        return num/len(acc_pre_index)
    else:
        total=0
        for tar in range(0,nb_classes):
            samples_path='../adv_result/'+datasets+'/'+attack+'/'+model_name+'/'+str(va)+'_'+str(tar)
            [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
            total+=len(image_list)
        return total/len(acc_pre_index)



def main(argv=None):


    result=mr(datasets='cifar10',
         model_name='vgg11',
         attack='cw',
         va=0.1)
    print(result)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    #flags.DEFINE_string('sample', '../datasets/integration/mnist/0.png', 'The path to load sample.')
    flags.DEFINE_string('model', 'lenet4', 'The name of model.')
    flags.DEFINE_string('attack', 'fgsm', 'step size of fgsm')

    tf.app.run()
