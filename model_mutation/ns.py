from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import math
import copy

sys.path.append("../")

from nmutant_model.model_operation import model_load, model_eval
from nmutant_data.data import get_data, get_shape
from nmutant_util.configs import path

FLAGS = flags.FLAGS


def ns(datasets, model_name, ration=0.1, threshold=0.9, batch_size=256, epoch=9):
    tf.reset_default_graph()
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    input_shape, nb_classes = get_shape(datasets)

    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, epoch=epoch)

    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params, feed=feed_dict)
    print('Test accuracy on legitimate test examples for original model: {0}'.format(accuracy))

    for i in range(len(model.layers)):
        layer = model.layers[i]
        if "Conv2D" in layer.__class__.__name__:
            unique_neurons_layer = layer.output_channels
            shuffle_num = unique_neurons_layer * ration
            if shuffle_num > 1.0:
                shuffle_num = math.floor(shuffle_num) if shuffle_num > 2.0 else math.ceil(shuffle_num)
                mutated_neurons = np.random.choice(unique_neurons_layer, int(shuffle_num), replace=False)
                current_weights = sess.run(layer.kernels).transpose([3,0,1,2])
                current_bias = sess.run(layer.b)
                shuffle_neurons = copy.copy(mutated_neurons)
                np.random.shuffle(shuffle_neurons)
                current_weights[mutated_neurons] = current_weights[shuffle_neurons]
                current_bias[mutated_neurons] = current_bias[shuffle_neurons]
                update_weights = tf.assign(layer.kernels, current_weights.transpose([1,2,3,0]))
                update_bias = tf.assign(layer.b, current_bias)
                sess.run(update_weights)
                sess.run(update_bias)
                if "BN" in model.layers[i + 1].__class__.__name__:
                    layer = model.layers[i + 1]
                    current_gamma = sess.run(layer.gamma)
                    current_beta = sess.run(layer.beta)
                    current_moving_mean = sess.run(layer.moving_mean)
                    current_moving_variance = sess.run(layer.moving_variance)
                    current_gamma[mutated_neurons] = current_gamma[shuffle_neurons]
                    current_beta[mutated_neurons] = current_beta[shuffle_neurons]
                    current_moving_mean[mutated_neurons] = current_moving_mean[shuffle_neurons]
                    current_moving_variance[mutated_neurons] = current_moving_variance[shuffle_neurons]
                    update_gamma = tf.assign(layer.gamma, current_gamma)
                    update_beta = tf.assign(layer.beta, current_beta)
                    update_moving_mean = tf.assign(layer.moving_mean, current_moving_mean)
                    update_moving_variance = tf.assign(layer.moving_variance, current_moving_variance)
                    sess.run(update_gamma)
                    sess.run(update_beta)
                    sess.run(update_moving_mean)
                    sess.run(update_moving_variance)
        elif "Linear" in layer.__class__.__name__ :
            unique_neurons_layer = layer.num_hid
            shuffle_num = unique_neurons_layer * ration
            if shuffle_num > 1.0:
                shuffle_num = math.floor(shuffle_num) if shuffle_num > 2.0 else math.ceil(shuffle_num)
                mutated_neurons = np.random.choice(unique_neurons_layer, int(shuffle_num), replace=False)
                current_weights = sess.run(layer.W).transpose([1,0])
                current_bias = sess.run(layer.b)
                shuffle_neurons = copy.copy(mutated_neurons)
                np.random.shuffle(shuffle_neurons)
                current_weights[mutated_neurons] = current_weights[shuffle_neurons]
                current_bias[mutated_neurons] = current_bias[shuffle_neurons]
                update_weights = tf.assign(layer.W, current_weights.transpose([1,0]))
                update_bias = tf.assign(layer.b, current_bias)
                sess.run(update_weights)
                sess.run(update_bias)

    mutated_accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params, feed=feed_dict)
    print('Test accuracy on legitimate test examples for mutated model: {0}'.format(mutated_accuracy))

    if mutated_accuracy >= threshold * accuracy:
        train_dir = os.path.join(path.mu_model_path, 'ns', datasets + '_' + model_name, '0')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        save_path = os.path.join(train_dir, datasets + '_' + model_name + '.model')
        saver = tf.train.Saver()
        saver.save(sess, save_path)

    sess.close()

def main(argv=None):
    ns(datasets=FLAGS.datasets,
        model_name=FLAGS.model,
        ration=FLAGS.ration,
        threshold=FLAGS.threshold)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet5', 'The name of model.')
    flags.DEFINE_float('ration', 0.1, 'The ration of mutated neurons.')
    flags.DEFINE_float('threshold', 0.9, 'The threshold of accuacy compared with original.')

    tf.app.run()