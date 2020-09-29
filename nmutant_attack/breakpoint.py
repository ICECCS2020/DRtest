"""
This tutorial shows how to generate adversarial examples
using JSMA in white-box setting.
The original paper can be found at:
https://arxiv.org/abs/1511.07528
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_model.model_operation import model_load
from nmutant_detection import utils

FLAGS = flags.FLAGS

def jsma(datasets,sample_path, model_path='../models/integration/mnist'):
    """
    the Jacobian-based saliency map approach (JSMA)
    :param datasets
    :param sample: inputs to attack
    :param target: the class want to generate
    :param nb_classes: number of output classes
    :return:
    """
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_path)

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    [X_test_adv, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(sample_path)
    import os
    for i in range(len(adv_image_files)):
        temp = adv_image_files[i].split('_')[-4]
        if os.path.exists("../datasets/integration/batch_attack/cifar10/" + str(temp) +'.png'):
            os.remove("../datasets/integration/batch_attack/cifar10/" + str(temp) +'.png')




def main(argv=None):
    jsma(datasets = FLAGS.datasets,
         sample_path=FLAGS.sample,
         model_path=FLAGS.model_path)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('sample', '../mt_result/integration/cw/cifar10', 'The path to load sample.')
    # flags.DEFINE_integer('target', 1, 'target')
    flags.DEFINE_string('model_path', '../models/integration/cifar10', 'The path to load model.')
    # flags.DEFINE_string('store_path', '../mt_result/integration/jsma/mnist', 'The path to store adversaries.')
    # flags.DEFINE_integer('nb_classes', 10, 'Number of output classes')

    tf.app.run()
