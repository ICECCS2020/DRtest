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
from nmutant_util.utils_tf import model_prediction, model_argmax
from nmutant_model.model_operation import model_load
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1
from nmutant_data.data import get_shape, get_data
import time
import math

FLAGS = flags.FLAGS

def jsma(datasets,sample, model_name, target,
         store_path, gamma=0.1, start=0, end=10000, batch_size=32, epoch=9, mu=False, mu_var='gf', de=False, attack='fgsm'):
    """
    the Jacobian-based saliency map approach (JSMA)
    :param datasets
    :param sample: inputs to attack
    :param target: the class want to generate
    :param nb_classes: number of output classes
    :return:
    """
    tf.reset_default_graph()
    X_train, Y_train, X_test, Y_test = get_data(datasets)

    # sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, epoch=epoch)
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, epoch=epoch, mu=mu, mu_var=mu_var, de=de, attack=attack)

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
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

    input_shape, nb_classes = get_shape(datasets)
    sample=sample[start:end]
    probabilities = model_prediction(sess, x, preds, sample, feed=feed_dict)
    current_class=[]
    for i in range(0, probabilities.shape[0]):
        current_class.append(np.argmax(probabilities[i])) 

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    '''
    if target == current_class:
        return 'The target is equal to its original class'
    elif target >= nb_classes or target < 0:
        return 'The target is out of range'
    '''
		
    #only for correct:
    Y_test=Y_test[start:end]
    acc_pre_index=[]
    for i in range(0, sample.shape[0]):
        if current_class[i]==np.argmax(Y_test[i]):
            acc_pre_index.append(i)

    print('Start generating adv. example for target class %i' % target)
    sample_acc=np.zeros(shape=(len(acc_pre_index),input_shape[1], input_shape[2], input_shape[3]), dtype='float')
    current_class_acc=np.zeros(shape=(len(acc_pre_index)), dtype=int)
    for i in range(0, len(acc_pre_index)):
        sample_acc[i]=sample[acc_pre_index[i]]
        current_class_acc[i]=current_class[acc_pre_index[i]]
    #print('current_class_acc',current_class_acc)
    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 1., 'gamma': gamma,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    # This call runs the Jacobian-based saliency map approach
    one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
    one_hot_target[0, target] = 1
    jsma_params['y_target'] = one_hot_target
	
    suc=0
    nb_batches = int(math.ceil(float(sample_acc.shape[0]) / batch_size))
    for batch in range(nb_batches): 
        #print(batch)
        start_batch=batch*batch_size
        end_batch=(batch+1)*batch_size
        if end_batch>sample_acc.shape[0]:
            end_batch=sample_acc.shape[0]
        adv_inputs=sample_acc[start_batch:end_batch]
        for j in range(start_batch, end_batch):
            if current_class_acc[j]!=target:
                adv_input=adv_inputs[j-start_batch].reshape(1,input_shape[1],input_shape[2],input_shape[3])
                adv = jsma.generate_np(adv_input, **jsma_params)
                new_class_labels = model_argmax(sess, x, preds, adv, feed=feed_dict)
                res = int(new_class_labels == target)
                if res == 1:
                    adv=adv.reshape(adv.shape[1],adv.shape[2],adv.shape[3])
                    #adv_img_deprocessed = deprocess_image_1(adv)
                    #adv_img_deprocessed=adv_img_deprocessed.reshape(adv_img_deprocessed.shape[1],adv_img_deprocessed.shape[2])
                    suc+=1
                    path = store_path + '/' + str(start+acc_pre_index[j]) + '_' + str(time.time()*1000) + '_' + str(current_class_acc[j]) + '_' + str(new_class_labels)
                    #path=store_path + '/'  + str(j)+ '_'+ str(current_class_acc[j]) +'.png'
                    #imsave(path, adv_img_deprocessed)
                    np.save(path, adv)
                    #print(adv.shape)

    # Close TF session
    sess.close()
    return suc, len(acc_pre_index)


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
