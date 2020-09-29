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

from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn
from tensorflow.python.training import moving_averages

class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):

    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                 keep_dims=True))
        with tf.name_scope("linear"):
            self.W = tf.Variable(init, name='kernel')
            self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'), name='bias')

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b
            
class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        # kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   # self.output_channels)
        if len(self.kernel_shape)==2:
            kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                    self.output_channels)
        else:
            kernel_shape = tuple(self.kernel_shape) + (self.output_channels,)

        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        with tf.name_scope("conv2d"):
            self.kernels = tf.Variable(init, name='kernel')
            self.b = tf.Variable(
                np.zeros((self.output_channels,)).astype('float32'), name='bias')
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b

class MaxPooling(Layer):

    def __init__(self, ksize, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        batch_size, rows, cols, input_channels = input_shape
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.max_pool(x, (1,) + tuple(self.ksize) + (1,), (1,) + tuple(self.strides) + (1,), self.padding)

class AvgPooling(Layer):

    def __init__(self, ksize, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        batch_size, rows, cols, input_channels = input_shape
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = batch_size
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.avg_pool(x, (1,) + tuple(self.ksize) + (1,), (1,) + tuple(self.strides) + (1,), self.padding)

class ReLU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.relu(x)

class Lrn(Layer):

    def __init__(self, depth_radius, bias, alpha, beta):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.lrn(x, depth_radius=self.depth_radius, bias=self.bias, alpha=self.alpha, beta=self.beta)

class Softmax(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)

class Dropout(Layer):

    def __init__(self, keep_prob):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.dropout(x, self.keep_prob)

class Flatten(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [shape[0], output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])

class Dense(Layer):

    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]

        self.kernel = vs.get_variable('kernel',
                                      shape=[input_shape[-1], self.num_hid],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                      dtype=dtypes.float32,
                                      trainable=True)
        self.bias = vs.get_variable('bias',
                                    shape=[self.num_hid, ],
                                    initializer=tf.zeros_initializer(),
                                    regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                    dtype=dtypes.float32,
                                    trainable=True)

    def fprop(self, x):
        if len(self.output_shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(x, self.kernel, [[len(self.input_shape) - 1],
                                                                   [0]])
            # Reshape the output back to the original ndim of the input.
            outputs.set_shape(self.output_shape)
        else:
            outputs = standard_ops.matmul(x, self.kernel)
        outputs = nn.bias_add(outputs, self.bias)
        # if self.activation is not None:
        #     return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

# class BN(Layer):
#     def __init__(self, epsilon=1e-03):
#         self.__dict__.update(locals())
#         del self.self
#
#     def set_input_shape(self, input_shape):
#         self.input_shape = input_shape
#         self.output_shape = input_shape
#         num_inputs = self.input_shape[-1]
#         self.reduce_dims = list(range(len(input_shape) - 1))
#         # tf.Variable(np.zeros((num_inputs,)).astype('float32'),name='beta')
#         self.beta = tf.Variable(np.zeros((num_inputs,)).astype('float32'),name='beta')
#         # self.beta = tf.get_variable("beta", shape=[num_inputs, ], dtype=tf.float32,
#         #                 initializer=tf.zeros_initializer())
#         self.gamma = tf.Variable(np.ones((num_inputs,)).astype('float32'),name="gamma")
#         # self.gamma = tf.get_variable("gamma", shape=[num_inputs, ], dtype=tf.float32,
#         #                             initializer=tf.ones_initializer())
#
#     def fprop(self, x):
#         mean, variance = tf.nn.moments(x, axes=self.reduce_dims)
#         return tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, self.epsilon)

# class BN(Layer):
#     def __init__(self, training):
#         self.__dict__.update(locals())
#         del self.self
#
#     def set_input_shape(self, input_shape):
#         self.input_shape = input_shape
#         self.output_shape = input_shape
#
#     def fprop(self, x):
#         return tf.layers.batch_normalization(x,training=self.training)

class BN(Layer):
    def __init__(self, training, decay=0.999, epsilon=1e-03):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        num_inputs = self.input_shape[-1]
        self.reduce_dims = list(range(len(input_shape) - 1))

        self.beta = tf.Variable(np.zeros((num_inputs,)).astype('float32'),name='beta')
        self.gamma = tf.Variable(np.ones((num_inputs,)).astype('float32'),name="gamma")

        # for inference
        self.moving_mean = tf.Variable(np.zeros((num_inputs,)).astype('float32'),name="moving_mean", trainable=False)
        self.moving_variance = tf.Variable(np.ones((num_inputs,)).astype('float32'),name="moving_variance", trainable=False)

    def fprop(self, x):
        if self.training:
            mean, variance = tf.nn.moments(x, axes=self.reduce_dims)
            update_move_mean = moving_averages.assign_moving_average(self.moving_mean,
                                                                     mean, decay=self.decay)
            update_move_variance = moving_averages.assign_moving_average(self.moving_variance,
                                                                         variance, decay=self.decay)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_move_variance)
        else:
            mean, variance = self.moving_mean, self.moving_variance
        return tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, self.epsilon)

# Temperature Scaling, "ENHANCING THE RELIABILITY OF OUT-OF-DISTRIBUTION IMAGE DETECTION IN NEURAL NETWORKS(ICLR2018)"
class TS(Layer):

    def __init__(self, ts):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.divide(x, self.ts)
