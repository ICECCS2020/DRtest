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

from nmutant_model.model_operation import model_load, model_eval
from nmutant_data.data import get_data, get_shape
from nmutant_util.configs import path

FLAGS = flags.FLAGS


def ws(datasets, model_name, ration=0.1, threshold=0.9, batch_size=256, epoch=9):
    tf.reset_default_graph()
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    input_shape, nb_classes = get_shape(datasets)

    sess, preds, x, y, model, feed_dict = model_load(datasets, model_name, epoch=epoch)

    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params, feed=feed_dict)
    print('Test accuracy on legitimate test examples for original model: {0}'.format(accuracy))

    unique_neurons = 0
    for layer in model.layers:
        if "Conv2D" in layer.__class__.__name__:
            unique_neurons += layer.output_channels
        elif "Linear" in layer.__class__.__name__ :
            unique_neurons += layer.num_hid
        # every BN neuron only connected with a previous neuron
    indices = np.random.choice(unique_neurons, int(unique_neurons * ration), replace=False)

    neurons_count = 0
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if "Conv2D" in layer.__class__.__name__:
            unique_neurons_layer = layer.output_channels
            mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
            if mutated_neurons:
                mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                current_weights = sess.run(layer.kernels).transpose([3,0,1,2])
                for neuron in mutated_neurons:
                    old_data = current_weights[neuron].reshape(-1)
                    shuffle_index = np.arange(len(old_data))
                    np.random.shuffle(shuffle_index)
                    new_data = old_data[shuffle_index].reshape(layer.kernels.shape[0], layer.kernels.shape[1], layer.kernels.shape[2])
                    current_weights[neuron] = new_data
                update_weights = tf.assign(layer.kernels, current_weights.transpose([1,2,3,0]))
                sess.run(update_weights)
            neurons_count += unique_neurons_layer
        elif "Linear" in layer.__class__.__name__ :
            unique_neurons_layer = layer.num_hid
            mutated_neurons = set(indices) & set(np.arange(neurons_count, neurons_count + unique_neurons_layer))
            if mutated_neurons:
                mutated_neurons = np.array(list(mutated_neurons)) - neurons_count
                current_weights = sess.run(layer.W).transpose([1,0])
                for neuron in mutated_neurons:
                    old_data = current_weights[neuron]
                    shuffle_index = np.arange(len(old_data))
                    np.random.shuffle(shuffle_index)
                    new_data = old_data[shuffle_index]
                    current_weights[neuron] = new_data
                update_weights = tf.assign(layer.W, current_weights.transpose([1,0]))
                sess.run(update_weights)
            neurons_count += unique_neurons_layer

    mutated_accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params, feed=feed_dict)
    print('Test accuracy on legitimate test examples for mutated model: {0}'.format(mutated_accuracy))

    if mutated_accuracy >= threshold * accuracy:
        train_dir = os.path.join(path.mu_model_path, 'ws', datasets + '_' + model_name, '0')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        save_path = os.path.join(train_dir, datasets + '_' + model_name + '.model')
        saver = tf.train.Saver()
        saver.save(sess, save_path)

    sess.close()

def main(argv=None):
    ws(datasets=FLAGS.datasets,
        model_name=FLAGS.model,
        ration=FLAGS.ration,
        threshold=FLAGS.threshold)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet5', 'The name of model.')
    flags.DEFINE_float('ration', 0.1, 'The ration of mutated neurons.')
    flags.DEFINE_float('threshold', 0.9, 'The threshold of accuacy compared with original.')

    tf.app.run()