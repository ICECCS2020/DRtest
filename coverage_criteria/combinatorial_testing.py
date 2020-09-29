#Combinatorial Testing for Deep Learning Systems, https://arxiv.org/abs/1806.07723#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import gc

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import math

sys.path.append("../")

from nmutant_model.model_operation import model_load
from coverage_criteria.utils import neuron_combination, cal_activation, update_combination
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_data.data import get_data
from nmutant_util.utils_file import get_data_file

FLAGS = flags.FLAGS

def ct(datasets, model_name, samples_path, t=2, p=0.5, de='False', attack='fgsm', just_adv=False, epoch=9):
    X_train, Y_train, X_test, Y_test = get_data(datasets)

    samples = X_test
    if samples_path not in ['test']:
        if not just_adv:
            [image_list, _, _, _] = get_data_file(samples_path) #image_files, real_labels, predicted_labels
            samples_adv = np.asarray(image_list)
            samples = np.concatenate((samples, samples_adv))
            print("Combine data")
        else:
            [image_list, _, _, _] = get_data_file(samples_path) #image_files, real_labels, predicted_labels
            samples = np.asarray(image_list)
            print("Just adv")

    tf.reset_default_graph()
    sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack,
                                                     epoch=epoch)
    layers_combination = neuron_combination(t, model, x)
    sess.close()
    del sess, preds, x, y, model, feed_dict
    gc.collect()

    n_batches = int(np.ceil(1.0 * samples.shape[0] / 512))
    for num in range(n_batches):
        print(num)
        start = num * 512
        end = np.minimum(len(samples), (num + 1) * 512)
        batch_samples = samples[start:end]
        tf.reset_default_graph()
        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack,
                                                         epoch=epoch)
        layers_activation = cal_activation(sess, x, batch_samples, model, feed_dict)
        sess.close()
        del sess, preds, x, y, model, feed_dict, batch_samples
        gc.collect()

        layers_combination = update_combination(layers_combination, layers_activation, t)

    sparse = 0
    dense = 0
    p_completeness = 0
    total = 0
    t_t = pow(2, t)
    completeness = 1.0 * t_t * p
    for layer in layers_combination:
        total += len(layer)
        for combination in layer:
            s = sum(combination)
            dense += s
            if s >= completeness:
                p_completeness += 1
            if combination == np.ones(t_t).astype('int').tolist():
                sparse += 1

    sparse_coverage = 1.0 * sparse / total
    dense_coverage = 1.0 * dense / (t_t * total)
    pt_completeness = 1.0 * p_completeness / total

    print([sparse_coverage, dense_coverage, pt_completeness])
    return [sparse_coverage, dense_coverage, pt_completeness]

def main(argv=None):
    ct(datasets = FLAGS.datasets,
       model_name=FLAGS.model,
       samples_path=FLAGS.samples,
       t=FLAGS.t,
       p=FLAGS.p)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet5', 'The name of model')
    flags.DEFINE_string('samples', 'test', 'The path to load samples.')  # '../mt_result/mnist_jsma/adv_jsma'
    flags.DEFINE_integer('t', 2,'t-way combination')
    flags.DEFINE_float('p', 0.5, 'p-completeness')
    tf.app.run()
