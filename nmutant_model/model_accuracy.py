import sys
sys.path.append('../')

from nmutant_data.cifar10 import data_cifar10
import tensorflow as tf
import numpy as np
from nmutant_data.data import get_shape, get_data
from nmutant_util.utils_imgproc import deprocess_image_1, preprocess_image_1, deprocess_image_1
from nmutant_util.utils_tf import model_prediction, model_argmax
from nmutant_model.model_operation import model_load
datasets = 'cifar10'
#model_name='vgg13'

X_train, Y_train, X_test, Y_test = get_data(datasets)
models=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
	     'resnet152', 'googlenet12', 'googlenet16', 'googlenet22']
sample=X_test[0:10000]
            
f=open('acc_cifar.txt' ,'w')

for model_name in models:
    tf.reset_default_graph()
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, de=False)
    print('load successfule')
    input_shape, nb_classes = get_shape(datasets)
    
    probabilities = model_prediction(sess, x, preds, sample, feed=feed_dict)
    current_class=[]
    for i in range(0, probabilities.shape[0]):
        current_class.append(np.argmax(probabilities[i])) 
    
    acc_pre_index=[]
    for i in range(0, sample.shape[0]):
        if current_class[i]==np.argmax(Y_test[i]):
            acc_pre_index.append(i)
    #print('current_class',current_class)
    f.write(model_name+' '+str(len(acc_pre_index)/X_test.shape[0]))
    f.write('\n')
f.close()
    
    
    