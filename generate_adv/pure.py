import sys
sys.path.append('../')
from tensorflow.python.platform import flags

from nmutant_model.model_operation import model_load
from nmutant_data.mnist import data_mnist
import tensorflow as tf
from nmutant_data.data import get_shape
from nmutant_util.utils_file import get_data_file
from nmutant_util.utils_tf import model_prediction
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1
import math
import os
from scipy.misc import imsave, imread
import numpy as np

FLAGS = flags.FLAGS
def pure(datasets='mnist', attack='fgsm', model_name='lenet1'):
    tf.reset_default_graph()
    samples_path='../adv_result/'+datasets+'/'+attack+'/'+model_name+'/pure'
    if not os.path.isdir(samples_path):
        os.makedirs(samples_path+'/train')
        os.makedirs(samples_path+'/test')
        
    samples_path_train='../adv_result/'+datasets+'/'+attack+'/'+model_name+'/train_data'
    samples_path_test='../adv_result/'+datasets+'/'+attack+'/'+model_name+'/test_data'
    
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name)
    
    [image_list_train, image_files_train, real_labels_train, predicted_labels_train] = get_data_file(samples_path_train)
    [image_list_test, image_files_test, real_labels_test, predicted_labels_test] = get_data_file(samples_path_test)

    #samples_train = np.asarray([preprocess_image_1(image.astype('float64')) for image in image_list_train])
    #samples_test = np.asarray([preprocess_image_1(image.astype('float64')) for image in image_list_test])
    samples_train = np.asarray(image_list_train)
    samples_test = np.asarray(image_list_test)
    
    probabilities_train = model_prediction(sess, x, preds, samples_train, feed=feed_dict)
    probabilities_test = model_prediction(sess, x, preds, samples_test, feed=feed_dict)

    for i in range(0, samples_train.shape[0]):
        if predicted_labels_train[i]==np.argmax(probabilities_train[i]):
            pure_train =samples_path+'/train/'+image_files_train[i]
            #imsave(pure_train, image_list_train[i])
            np.save(pure_train, image_list_train[i])

    for i in range(0, samples_test.shape[0]):
        if predicted_labels_test[i]==np.argmax(probabilities_test[i]):
            pure_test =samples_path+'/test/'+image_files_test[i]
            #imsave(pure_test, image_list_test[i])
            np.save(pure_test, image_list_test[i])
            
def main(argv=None):
    datasets='cifar10'
    attacks=['cw']
    model_names=['resnet101']
    for attack in attacks:
        for model_name in model_names:
            pure(datasets=datasets, attack=attack, model_name=model_name)
    
    #choose_test(datasets = FLAGS.datasets,
    #                attack=FLAGS.attack,
    #                model_name=FLAGS.model_name)
	

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('attack', 'cw', 'attack_method')#'../mt_result/mnist_jsma/adv_jsma'
    flags.DEFINE_string('model_name', 'resnet101', 'model_name')

    tf.app.run()


			
			
			
			

