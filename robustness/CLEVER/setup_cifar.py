## setup_cifar.py -- cifar data and model loading code
##
## Copyright (C) 2017-2018, IBM Corp.
## Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
##                and Huan Zhang <ecezhang@ucdavis.edu>
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the Apache 2.0 licence,
## contained in the LICENCE file in this directory.


import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K

import sys

sys.path.append('../../')
from nmutant_data.data import get_data
from nmutant_data.data import get_shape
from nmutant_model.tutorial_models import sub_model, VGG11, VGG13, VGG16, VGG19, VGG_test, LeNet_1, LeNet_4, LeNet_5, ResNet18, ResNet34, ResNet50, ResNet101,                                                 ResNet152, GoogleNet12, GoogleNet16, GoogleNet22
model_dict = {"sub":sub_model,
              "vgg11":VGG11, "vgg13":VGG13, "vgg16":VGG16, "vgg19":VGG19, "vgg_test":VGG_test,
              "lenet1":LeNet_1, "lenet4":LeNet_4, "lenet5":LeNet_5,
              "resnet18":ResNet18, "resnet34":ResNet34, "resnet50":ResNet50, "resnet101":ResNet101, "resnet152":ResNet152,
              "googlenet12":GoogleNet12, "googlenet16":GoogleNet16, "googlenet22":GoogleNet22}

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)
    

class CIFAR:
    def __init__(self):

        train_data, train_labels, self.test_data, self.test_labels = get_data('cifar10')
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]

class CIFARModel:
    def __init__(self, restore=None, session=None, use_softmax=False, use_brelu = False, activation = "relu", de=False, attack='fgsm', epoch=49):
        def bounded_relu(x):
                return K.relu(x, max_value=1)
        if use_brelu:
            activation = bounded_relu
        else:
            activation = activation

        print("inside CIFARModel: activation = {}".format(activation))

        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        x=tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.float32, shape=(None, 10))
        
        input_shape, nb_classes = get_shape('cifar10')

        model=model_dict[restore](input_shape, nb_classes, False)
        preds=model(x)
        if de==True:
            model_path = '../../de_models/'+attack+'/' + 'cifar10_'+restore+'/'+str(epoch)+'/'+restore+'.model'
        else:
            model_path = '/mnt/dyz/models/'+'cifar10_'+restore+'/'+str(epoch)+'/'+restore+'.model'
        saver = tf.train.Saver()
        saver.restore(session, model_path)
        print("load model successfully")
        layer_outputs = []
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        self.layer_outputs = layer_outputs
        self.model = model

    def predict(self, data):
        return self.model(data)
        
    
class TwoLayerCIFARModel:
    def __init__(self, restore = None, session=None, use_softmax=False):
        self.num_channels = 3
        self.image_size = 32
        self.num_labels = 10

        model = Sequential()
        model.add(Flatten(input_shape=(32, 32, 3)))
        model.add(Dense(1024))
        model.add(Activation('softplus'))
        model.add(Dense(10))
        # output log probability, used for black-box attack
        if use_softmax:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        layer_outputs = []
        for layer in model.layers:
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))

        self.layer_outputs = layer_outputs
        self.model = model

    def predict(self, data):

        return self.model(data)
