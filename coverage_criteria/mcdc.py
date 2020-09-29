#https://arxiv.org/pdf/1805.00089.pdf#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import math
import gc

sys.path.append("../")

from nmutant_model.model_operation import model_load
from coverage_criteria.utils import cal_sign, xor
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_data.data import get_data
from nmutant_util.utils_file import get_data_file

FLAGS = flags.FLAGS

def mcdc(datasets, model_name, samples_path, de='False', attack='fgsm', just_adv=False, epoch=9):
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

    tf.reset_default_graph()
    sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack,
                                                     epoch=epoch)
    # dict = model.fprop(x)

    layer_names = model.layer_names
    sess.close()
    del sess, preds, x, y, model, feed_dict
    gc.collect()

    l = 0
    ss = []
    sc_pr = []
    neuron = []
    for key in layer_names:#model.layer_names:
        if 'ReLU' in key or 'probs' in key:
            print(l)
            tf.reset_default_graph()
            sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack,
                                                             epoch=epoch)
            dict = model.fprop(x)

            tensor = dict[key]
            neuron.append(tensor.shape[-1])
            layer_output = []
            n_batches = int(np.ceil(1.0 * samples.shape[0] / 256))
            for batch in range(n_batches):
                start = batch * 256
                end = np.minimum(len(samples), (batch + 1) * 256)
                feed = {x: samples[start:end]}
                if feed_dict is not None:
                    feed.update(feed_dict)
                # v = sess.run(tensor, feed_dict=feed)
                layer_output = layer_output + sess.run(tensor, feed_dict=feed).tolist()

            sess.close()
            del sess, preds, x, y, model, feed_dict
            gc.collect()

            layer_output = np.asarray(layer_output)
            layer_sign = np.zeros((layer_output.shape[0], layer_output.shape[-1]))
            for num in range(len(layer_output)):
                for num_neuron in xrange(layer_output[num].shape[-1]):
                    if np.mean(layer_output[num][..., num_neuron]) > 0.0:
                        layer_sign[num][num_neuron] = 1

            del layer_output
            gc.collect()

            if l == 0:
                for i in range(len(samples) - 1):
                    temp = []
                    for j in range(i + 1, len(samples)):
                        sc = xor(layer_sign[i], layer_sign[j])
                        if len(sc) > 1:
                            sc = []
                        temp.append(sc)
                    sc_pr.append(temp)
            else:
                for i in range(len(samples) - 1):
                    for j in range(i+1, len(samples)):
                        sc = xor(layer_sign[i], layer_sign[j])
                        if len(sc_pr[i][j-i-1]) == 1:
                            n_pr = str(l) + '_' + str(sc_pr[i][j-i-1])
                            for n in sc:
                                n = str(l + 1) + '_' + str(n)
                                combination = tuple([n_pr, n])
                                if combination not in ss:
                                    # ss.append(tuple([n_pr, n]))
                                    ss.append(combination)
                        if len(sc) > 1:
                            sc = []
                        sc_pr[i][j-i-1] = sc
            l = l+1
    total = 0
    for i in range(len(neuron) - 1):
        total = total + int(neuron[i]) * int(neuron[i+1])

    print(1.0 * len(set(ss)) / total)
    return 1.0 * len(set(ss)) / total

def main(argv=None):
    mcdc(datasets = FLAGS.datasets,
         model_name=FLAGS.model,
         samples_path=FLAGS.samples)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet5', 'The name of model')
    flags.DEFINE_string('samples', 'test', 'The path to load samples.')  # '../mt_result/mnist_jsma/adv_jsma'

    tf.app.run()

