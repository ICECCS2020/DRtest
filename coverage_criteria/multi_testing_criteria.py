#Ma, Lei, et al. "DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems." (2018):120-131.#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import gc

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import math

sys.path.append("../")

from nmutant_model.model_operation import model_load
from coverage_criteria.utils import neuron_boundary, calculate_layers, update_multi_coverage_neuron, calculate_coverage_layer, init_coverage_metric
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_data.data import get_data
from nmutant_util.utils_file import get_data_file

FLAGS = flags.FLAGS

def multi_testing_criteria(datasets, model_name, samples_path, std_range = 0.0, k_n = 1000, k_l = 2, de=False, attack='fgsm', just_adv=False, epoch=9):
    """
    :param datasets
    :param model
    :param samples_path
    :param std_range
    :param k_n
    :param k_l
    :return:
    """
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    samples = X_test
    if samples_path not in ['test']:
        if not just_adv:
            [image_list, _, _, _] = get_data_file(samples_path)
            samples_adv = np.asarray(image_list)
            samples = np.concatenate((samples, samples_adv))
            print("Combine data")
        else:
            [image_list, _, _, _] = get_data_file(samples_path)
            samples = np.asarray(image_list)
            print("Just adv")

    if de == True:
        train_boundary_path = '../adv_result/' + datasets + '/' + attack + '/' + model_name + '/train_data'
        [image_list_train, _, _, _] = get_data_file(train_boundary_path)
        samples_train = np.asarray(image_list_train)

        X_train_boundary = np.concatenate((samples_train, X_train))
        store_path = "../multi_testing_criteria/" + datasets + "/" + model_name + "/" + attack + '/'
    else:
        X_train_boundary = X_train
        store_path = "../multi_testing_criteria/" + datasets + "/" + model_name + "/ori/"

    if not os.path.exists(store_path):
        os.makedirs(store_path)
        tf.reset_default_graph()
        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack,
                                                         epoch=epoch)
        boundary = neuron_boundary(sess, x, X_train_boundary, model, feed_dict)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()
        np.save(store_path + "boundary.npy", np.asarray(boundary))
    else:
        boundary = np.load(store_path + "boundary.npy").tolist()

    k_coverage, boundary_coverage, neuron_number = init_coverage_metric(boundary, k_n)

    if samples_path == 'test':
        store_path = store_path + 'test/'
    else:
        store_path = store_path + samples_path.split('/')[-3] + '/'

    if not os.path.exists(store_path):
        cal = True
        os.makedirs(store_path)
    else:
        cal = False

    NP = []
    n_batches = int(np.ceil(1.0 * samples.shape[0] / 256))
    for num in range(n_batches):
        print(num)
        start = num * 256
        end = np.minimum(len(samples), (num + 1) * 256)
        if cal:
            input_data = samples[start:end]
            tf.reset_default_graph()
            sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de,
                                                             attack=attack, epoch=epoch)
            layers_output = calculate_layers(sess, x, model, feed_dict, input_data, store_path, num)

            sess.close()
            del sess, preds, x, y, model, feed_dict, input_data
            gc.collect()
        else:
            layers_output = np.load(store_path + 'layers_output_' + str(num) + '.npy')

        k_coverage, boundary_coverage = update_multi_coverage_neuron(layers_output, k_n, boundary, k_coverage, boundary_coverage, std_range)

        layer_coverage = calculate_coverage_layer(layers_output, k_l, end - start)

        if num == 0:
            layer = [set([])] * layer_coverage.shape[0]
        for i in range(len(layer_coverage)):
            for j in range(len(layer_coverage[i])):
                layer[i] = layer[i] | layer_coverage[i][j]

        sample_coverage = np.transpose(layer_coverage, (1, 0))
        for i in range(len(sample_coverage)):
            sc = sample_coverage[i].tolist()
            if sc not in NP:
                NP.append(sc)

        del layers_output
        gc.collect()

    KMN = 0
    NB = 0
    SNA = 0
    for i in range(len(k_coverage)):
        for j in range(len(k_coverage[i])):
            for t in range(len(k_coverage[i][j])):
                if k_coverage[i][j][t] > 0:
                    KMN += 1
            if boundary_coverage[i][j][1] > 0:
                NB += 1
                SNA += 1
            if boundary_coverage[i][j][0] > 0:
                NB += 1
    KMN = 1.0 * KMN / (k_n * neuron_number)
    NB = 1.0 * NB / (2 * neuron_number)
    SNA = 1.0 * SNA / neuron_number

    TKNC = sum(len(neurons) for neurons in layer)
    TKNC = 1.0 * TKNC / neuron_number

    TKNP = len(NP)

    print([KMN, NB, SNA, TKNC, TKNP])
    return [KMN, NB, SNA, TKNC, TKNP]

def main(argv=None):
    multi_testing_criteria(datasets = FLAGS.datasets,
                           model_name=FLAGS.model,
                           samples_path=FLAGS.samples,
                           std_range = FLAGS.std_range,
                           k_n = FLAGS.k_n,
                           k_l = FLAGS.k_l)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model')
    flags.DEFINE_string('samples', 'test', 'The path to load samples.')  # '../mt_result/mnist_jsma/adv_jsma'
    flags.DEFINE_float('std_range', 0.0, 'The parameter to difine boundary with std')
    flags.DEFINE_integer('k_n', 1000, 'The number of sections for neuron output')
    flags.DEFINE_integer('k_l', 2, 'The number of top-k neurons in one layer')

    tf.app.run()
