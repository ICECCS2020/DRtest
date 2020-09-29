## setup_mnist.py -- mnist data and model loading code
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
import urllib.request

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
import sys

sys.path.append('../../')
from nmutant_data.data import get_data, get_shape
from nmutant_model.tutorial_models import sub_model, VGG11, VGG13, VGG16, VGG19, VGG_test, LeNet_1, LeNet_4, LeNet_5, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, GoogleNet12, GoogleNet16, GoogleNet22
model_dict = {"sub":sub_model,
              "vgg11":VGG11, "vgg13":VGG13, "vgg16":VGG16, "vgg19":VGG19, "vgg_test":VGG_test,
              "lenet1":LeNet_1, "lenet4":LeNet_4, "lenet5":LeNet_5,
              "resnet18":ResNet18, "resnet34":ResNet34, "resnet50":ResNet50, "resnet101":ResNet101, "resnet152":ResNet152,
              "googlenet12":GoogleNet12, "googlenet16":GoogleNet16, "googlenet22":GoogleNet22}


def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data, train_labels, self.test_data, self.test_labels = get_data('mnist')
        #train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        #train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        #self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        #self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class MNISTModel:
    def __init__(self, restore = None, session=None, use_softmax=False, use_brelu = False, activation = "relu", de=False, attack='fgsm', epoch=49):
        def bounded_relu(x):
                return K.relu(x, max_value=1)
        if use_brelu:
            activation = bounded_relu

        print("inside MNISTModel: activation = {}".format(activation))

        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10
        x=tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
        y = tf.placeholder(tf.float32, shape=(None, 10))
        input_shape, nb_classes = get_shape('mnist')
        model=model_dict[restore](input_shape, nb_classes, False)
        preds=model(x)
        if de==True:
            model_path = '../../de_models/'+attack+'/' + 'mnist_'+restore+'/'+str(epoch)+'/'+restore+'.model'
        else:
            model_path = '/mnt/dyz/models/'+'mnist_'+restore+'/'+str(epoch)+'/'+restore+'.model'
            #model_path='../../models/'+mnist_'+restore+'.model'
        saver = tf.train.Saver()
        saver.restore(session, model_path)
        print("load model successfully")
        '''
        layer_outputs = []
        for layer in model.layers:
            print(layer)
            if isinstance(layer, Conv2D) or isinstance(layer, Dense):
                layer_outputs.append(K.function([model.layers[0].input], [layer.output]))
        '''
        self.model = model
        #self.layer_outputs = layer_outputs
        

    def predict(self, data):
        return self.model(data)

class TwoLayerMNISTModel:
    def __init__(self, restore = None, session=None, use_softmax=False):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(1024))
        model.add(Lambda(lambda x: x * 10))
        model.add(Activation('softplus'))
        model.add(Lambda(lambda x: x * 0.1))
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

