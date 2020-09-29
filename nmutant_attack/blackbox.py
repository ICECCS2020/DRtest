"""
This tutorial shows how to generate adversarial examples
using FGSM in black-box setting.
The original paper can be found at:
https://arxiv.org/abs/1602.02697
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")
from scipy.misc import imsave, imread
import time

from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_data.svhn import data_svhn
from nmutant_util.utils_tf import model_argmax, model_prediction
from nmutant_model.model_operation import model_load, sub_model_load
from nmutant_attack.attacks import FastGradientMethod
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1
from nmutant_data.data import get_data, get_shape
import time
import math

FLAGS = flags.FLAGS

def blackbox(datasets, sample, model_name, submodel_name,
             store_path, step_size=0.3, batch_size=256):
    """
    the black-box attack from arxiv.org/abs/1602.02697
    :param datasets
    :param sample: inputs to attack
    :param target: the class want to generate
    :param nb_classes: number of output classes
    :return:
    """
    # Simulate the black-box model locally
    # You could replace this by a remote labeling API for instance
    print("Preparing the black-box model.")
    tf.reset_default_graph()
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    input_shape, nb_classes = get_shape(datasets)
    sess, bbox_preds, x, y, model, feed_dict = model_load(datasets, model_name)

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Preparing the substitute model.")
    model_sub, preds_sub = sub_model_load(sess, datasets, submodel_name, model_name)

    ###########################################################################
    # Craft adversarial examples using the Blackbox approach
    ###########################################################################
    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    '''
    if 'mnist' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(28,28,1)]).astype('float32')
        sample = preprocess_image_1(sample)
    elif 'cifar10' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(32,32,3)]).astype('float32')
        sample = preprocess_image_1(sample)
    elif 'svhn' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(32,32,3)]).astype('float32')
        sample = preprocess_image_1(sample)
    '''
	
    probabilities = model_prediction(sess, x, model(x), sample, feed=feed_dict)
    if sample.shape[0] == 1:
        current_class = np.argmax(probabilities)
    else:
        current_class = np.argmax(probabilities, axis=1)

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # only for correct:
    acc_pre_index=[]
    for i in range(0, sample.shape[0]):
        if current_class[i]==np.argmax(Y_test[i]):
            acc_pre_index.append(i)

    sample_acc=np.zeros(shape=(len(acc_pre_index),input_shape[1], input_shape[2], input_shape[3]), dtype='float32')
    probabilities_acc=np.zeros(shape=(len(acc_pre_index),nb_classes), dtype='float32')
    current_class_acc=np.zeros(shape=(len(acc_pre_index)), dtype=int)

    for i in range(0, len(acc_pre_index)):
        sample_acc[i]=sample[acc_pre_index[i]]
        probabilities_acc[i]=probabilities[acc_pre_index[i]]
        current_class_acc[i]=current_class[acc_pre_index[i]]
		
		
    if datasets == 'mnist':
        fgsm_par = {'eps': step_size, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    elif 'cifar10' == datasets:
        fgsm_par = {'eps': step_size, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    elif 'svhn' == datasets:
        fgsm_par = {'eps': step_size, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute
    x_adv_sub = fgsm.generate(x, **fgsm_par)

    nb_batches = int(math.ceil(float(sample_acc.shape[0]) / batch_size))
    suc=0
    for batch in range(nb_batches):
        #start, end = batch_indices(batch, sample_acc.shape[0], batch_size)
        print(batch)
        start=batch*batch_size
        end=(batch+1)*batch_size
        if end>sample_acc.shape[0]:
            end=sample_acc.shape[0]
        adv= sess.run(x_adv_sub,feed_dict={x: sample_acc[start:end], y: probabilities_acc[start:end]})
        adv_img_deprocessed = deprocess_image_1(adv)

    # Check if success was achieved
    #probabilities = model_prediction(sess, x, preds, sample, feed=feed_dict)
        new_class_label = model_argmax(sess, x, model(x), adv, feed=feed_dict)  # Predicted class of the generated adversary
        for i in range(0, len(new_class_label)):
            j=batch*batch_size+i
            if new_class_label[i]!=current_class_acc[j]:
                suc+=1
                path = store_path + '/' + str(j) + '_' + str(time.time()*1000) + '_' + str(current_class_acc[j]) + '_' + str(new_class_label[i]) + '.png'
                imsave(path, adv_img_deprocessed[i])
    # Close TF session   
    sess.close()
  

    return suc, len(acc_pre_index)
	
    '''
    adv = sess.run(x_adv_sub, feed_dict={x: sample, y: probabilities})

    new_class_label = model_argmax(sess, x, model(x), adv, feed=feed_dict)
    res = int(new_class_label != current_class)

    if res == 1:
        adv_img_deprocessed = deprocess_image_1(adv)
        i = sample_path.split('/')[-1].split('.')[-2]
        path = store_path + '/adv_' + str(time.time()*1000) + '_' + i + '_' + str(current_class) + '_' + str(new_class_label) + '_.png'
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
        sample = X_test[2:3]
        imsave(FLAGS.sample, deprocess_image_1(sample))
    elif 'cifar10' == datasets:
        train_start = 0
        train_end = 50000
        test_start = 0
        test_end = 10000

        # Get CIFAR10 test data
        X_train, Y_train, fn_train, X_test, Y_test, fn_test = data_cifar10(train_start=train_start,
                                                                           train_end=train_end,
                                                                           test_start=test_start,
                                                                           test_end=test_end,
                                                                           preprocess=preprocess_image_1)
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
    blackbox(datasets=FLAGS.datasets,
             sample_path=FLAGS.sample,
             model_name=FLAGS.model,
             submodel_name=FLAGS.submodel,
             store_path=FLAGS.store_path)


if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('sample', '../datasets/integration/mnist/2.png', 'The path to load sample.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model.')
    flags.DEFINE_string('submodel', 'lenet1', 'The name of submodel.')
    flags.DEFINE_string('store_path', '../mt_result/integration/blackbox/mnist', 'The path to store adversaries.')

    tf.app.run()

