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

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from input_mutation import utils

from nmutant_util.utils_tf import model_argmax
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_model.model_operation import model_load

FLAGS = flags.FLAGS

def mutation_tutorial(datasets, attack, sample_path, store_path, model_path, level=1, test_num=100, mutation_number=1000, mutated=False):
    sess, preds, x, y, model, feed_dict = model_load(datasets, model_path + datasets)

    sample_path = sample_path + attack + '/' + datasets
    # sample_path = '../mt_result/mnist_jsma/adv_jsma'
    [image_list, image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(sample_path)
    count = 0
    for i in range(len(image_list)):
        ori_img = preprocess_image_1(image_list[i].astype('float64'))
        ori_img = np.expand_dims(ori_img.copy(), 0)
        p = model_argmax(sess, x, preds, ori_img, feed=feed_dict)
        if p != predicted_labels[i]:
            count = count + 1
            image_file = image_files[i]
            os.remove("../datasets/adversary/" + attack + '/' + datasets + '/' + image_file)
            # os.remove(sample_path + '/' + image_file)

    # Close TF session
    print(count)
    sess.close()
    print('Finish.')


def main(argv=None):
    mutation_tutorial(datasets=FLAGS.datasets,
                      attack=FLAGS.attack,
                      sample_path=FLAGS.sample_path,
                      store_path=FLAGS.store_path,
                      model_path=FLAGS.model_path,
                      level=FLAGS.level,
                      test_num=FLAGS.test_num,
                      mutation_number=FLAGS.mutation_num,
                      mutated=FLAGS.mutated)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('attack', 'fgsm', 'The type of generating adversaries')
    flags.DEFINE_boolean('mutated', False, 'The mutation list is already generate.')  # default:False
    flags.DEFINE_string('model_path', '../models/das_cnn/','Path to save or load the model file')
    flags.DEFINE_string('store_path', '../results/', 'The path to store results.')
    flags.DEFINE_string('sample_path', '../datasets/adversary/', 'The path to load adversaries.')
    flags.DEFINE_integer('level', 1, 'the level of random mutation region.')
    flags.DEFINE_integer('test_num', 100, 'Number of mutation test targets')
    flags.DEFINE_integer('mutation_num', 500, 'Number of mutation tests')

    tf.app.run()
