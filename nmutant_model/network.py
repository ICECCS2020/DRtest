"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from nmutant_model.model import Model
from nmutant_model.layer import *

class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
                layer.name = name
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states

class ResidualModel(Model):

    def __init__(self, layers, input_shape):
        self.layer_names = []
        # self.layers = layers
        self.layers = []
        self.layers_list = layers
        self.input_shape = input_shape

        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers_list):
            if isinstance(layer, Layer):
                self.layers.append(layer)
                if hasattr(layer, 'name'):
                    name = layer.name
                else:
                    name = layer.__class__.__name__ + str(i)
                    layer.name = name
                self.layer_names.append(name)

                layer.set_input_shape(input_shape)
                input_shape = layer.get_output_shape()
            else:
                output_shape = None
                for j, l in enumerate(layer):
                    if isinstance(l, Layer):
                        self.layers.append(l)
                        if hasattr(l, 'name'):
                            name = l.name
                        else:
                            name = l.__class__.__name__ + str(i) + '_' + str(j)
                            l.name = name
                        self.layer_names.append(name)

                        l.set_input_shape(output_shape)
                        input_shape = l.get_output_shape()
                    else:
                        sl_input = input_shape
                        for k, sl in enumerate(l):
                            self.layers.append(sl)
                            if hasattr(sl, 'name'):
                                name = sl.name
                            else:
                                name = sl.__class__.__name__ + str(i) + '_' + str(j) + '_' + str(k)
                                sl.name = name
                            self.layer_names.append(name)

                            sl.set_input_shape(sl_input)
                            sl_input = sl.get_output_shape()
                            output_shape = sl_input

    def fprop(self, x):
        states = []
        for layer in self.layers_list:
            if isinstance(layer, Layer):
                x = layer.fprop(x)
                assert x is not None
                states.append(x)
            else:
                temp = []
                for l in layer:
                    if isinstance(l, Layer):
                        x = temp[0] + temp[1]
                        x = l.fprop(x)
                        assert x is not None
                        states.append(x)
                        temp.append(x)
                    else:
                        t_x = x
                        for sl in l:
                            t_x = sl.fprop(t_x)
                            assert t_x is not None
                            states.append(t_x)
                        temp.append(t_x)
        states = dict(zip(self.get_layer_names(), states))
        return states

class EnsembleModel(Model):

    def __init__(self, layers, input_shape):
        self.layer_names = []
        # self.layers = layers
        self.layers = []
        self.layers_list = layers
        self.input_shape = input_shape

        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers_list):
            if isinstance(layer, Layer):
                self.layers.append(layer)
                if hasattr(layer, 'name'):
                    name = layer.name
                else:
                    name = layer.__class__.__name__ + str(i)
                    layer.name = name
                self.layer_names.append(name)

                layer.set_input_shape(input_shape)
                input_shape = layer.get_output_shape()
            else:
                last = 0
                for j, l in enumerate(layer):
                    if isinstance(l, Layer):
                        self.layers.append(l)
                        if hasattr(l, 'name'):
                            name = l.name
                        else:
                            name = l.__class__.__name__ + str(i) + '_' + str(j)
                            l.name = name
                        self.layer_names.append(name)

                        l.set_input_shape(input_shape)
                        output_shape = l.get_output_shape()
                        last += output_shape[-1]
                    else:
                        s_input = input_shape
                        for k, sl in enumerate(l):
                            self.layers.append(sl)
                            if hasattr(sl, 'name'):
                                name = sl.name
                            else:
                                name = sl.__class__.__name__ + str(i) + '_' + str(j) + '_' + str(k)
                                sl.name = name
                            self.layer_names.append(name)

                            sl.set_input_shape(s_input)
                            s_input = sl.get_output_shape()

                        output_shape = s_input
                        last += output_shape[-1]

                output_shape = list(output_shape)
                output_shape[-1] = last
                input_shape = tuple(output_shape)

    def fprop(self, x):
        states = []
        for layer in self.layers_list:
            if isinstance(layer, Layer):
                x = layer.fprop(x)
                assert x is not None
                states.append(x)
            else:
                temp = []
                for l in layer:
                    if isinstance(l, Layer):
                        t_x = l.fprop(x)
                        assert t_x is not None
                        states.append(t_x)
                        temp.append(t_x)
                    else:
                        t_x = x
                        for sl in l:
                            t_x = sl.fprop(t_x)
                            assert t_x is not None
                            states.append(t_x)
                        temp.append(t_x)
                x = tf.concat(temp, 3)
        states = dict(zip(self.get_layer_names(), states))
        return states
