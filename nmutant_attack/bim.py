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
from nmutant_attack.attacks import BasicIterativeMethod
from nmutant_util.utils_imgproc import preprocess_image_1, deprocess_image_1
from nmutant_data.data import get_data, get_shape
from nmutant_util.utils import batch_indices
import time
import math

FLAGS = flags.FLAGS
    
def bim(datasets, sample, model_name, store_path, step_size='0.3', batch_size=256, epoch=9):
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

    print(epoch)
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, epoch=epoch)

    ###########################################################################
    # Craft adversarial examples using the BIM approach
    ###########################################################################
    # Initialize the Basic Iterative Method (BIM) attack object and
    # graph
    '''
    if 'mnist' == datasets:
        #sample = np.asarray([np.asarray(imread(sample_path)).reshape(28,28,1)]).astype('float32')
        #sample = preprocess_image_1(sample)
        print('1')
    elif 'cifar10' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(32,32,3)]).astype('float32')
        sample = preprocess_image_1(sample)
    elif 'svhn' == datasets:
        sample = np.asarray([np.asarray(imread(sample_path)).reshape(32,32,3)]).astype('float32')
        sample = preprocess_image_1(sample)
    #print(sample.shape)
    '''

    probabilities = model_prediction(sess, x, preds, sample, feed=feed_dict)

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
    print('Start generating adv. example')
    #print(float(step_size))
    if 'mnist' == datasets:
        bim_params = {'eps': float(step_size),
                      'eps_iter': float(step_size) / 6,
                      'clip_min': 0.,
                      'clip_max': 1.}
    elif 'cifar10' == datasets:
        bim_params = {'eps': float(step_size),
                      'eps_iter': float(step_size) / 6,
                      'clip_min': 0.,
                      'clip_max': 1.}
    elif 'svhn' == datasets:
        bim_params = {'eps': float(step_size),
                      'eps_iter': float(step_size) / 6,
                      'clip_min': 0.,
                      'clip_max': 1.}
    bim = BasicIterativeMethod(model, sess=sess)
    adv_x = bim.generate(x, **bim_params)
    
    nb_batches = int(math.ceil(float(sample_acc.shape[0]) / batch_size))
    suc=0
    for batch in range(nb_batches):
        #start, end = batch_indices(batch, sample_acc.shape[0], batch_size)
        print(batch)
        start=batch*batch_size
        end=(batch+1)*batch_size
        if end>sample_acc.shape[0]:
            end=sample_acc.shape[0]
        adv= sess.run(adv_x,feed_dict={x: sample_acc[start:end], y: probabilities_acc[start:end]})
        
        #adv_img_deprocessed = deprocess_image_1(adv)
    #adv:float 0-1 numpy.save("filename.npy",a)
    
    # Check if success was achieved
    #probabilities = model_prediction(sess, x, preds, sample, feed=feed_dict)
        new_class_label = model_argmax(sess, x, preds, adv, feed=feed_dict)  # Predicted class of the generated adversary
        for i in range(0, len(new_class_label)):
            j=batch*batch_size+i
            if new_class_label[i]!=current_class_acc[j]:
                suc+=1
                path = store_path + '/' + str(acc_pre_index[j]) + '_' + str(time.time()*1000) + '_' + str(current_class_acc[j]) + '_' + str(new_class_label[i])
                np.save(path,adv[i])
                # adv_img_deprocessed = deprocess_image_1(adv[i:i+1])
                # adv_img_deprocessed=adv_img_deprocessed.reshape(adv_img_deprocessed.shape[1],adv_img_deprocessed.shape[2])
                # path =  store_path + '/' + str(acc_pre_index[j]) + '_' + str(time.time()*1000) + '_' + str(current_class_acc[j]) + '_' + str(new_class_label[i])+'.png'
                #print(adv[i].shape)
                # imsave(path, adv_img_deprocessed)
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
        sample = X_test[0:10]

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
        sample = X_test[0:2]
        #imsave('a.png', deprocess_image_3(sample))
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
    
    store_path = 'test0.03'

    bim(datasets=datasets,
         sample=sample,
         model_name=FLAGS.model,
         store_path=store_path,
         step_size='0.03')


if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    #flags.DEFINE_string('sample', '../datasets/integration/mnist/0.png', 'The path to load sample.')
    flags.DEFINE_string('model', 'vgg16', 'The name of model.')
    flags.DEFINE_string('step_size', '0.02', 'step size of fgsm')
    #flags.DEFINE_string('store_path', '../adv_result/mnist/fgsm', 'The path to store adversaries.')

    tf.app.run()
