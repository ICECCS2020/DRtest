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
from input_mutation.mutation import MutationTest

from nmutant_util.utils_tf import model_argmax
from nmutant_util.utils_imgproc import preprocess_image_1
from nmutant_model.model_operation import model_load
from nmutant_data.mnist import data_mnist
from nmutant_data.cifar10 import data_cifar10
from nmutant_data.svhn import data_svhn

FLAGS = flags.FLAGS

def mutation_tutorial(datasets, attack, sample_path, store_path, model_path, level=1, test_num=100, mutation_number=1000, mutated=False):
    if 'mnist' == datasets:
        train_start = 0
        train_end = 60000
        test_start = 0
        test_end = 10000

        # Get MNIST test data
        X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                      train_end=train_end,
                                                      test_start=test_start,
                                                      test_end=test_end)
    elif 'cifar10' == datasets:
        preprocess_image = preprocess_image_1
        train_start = 0
        train_end = 50000
        test_start = 0
        test_end = 10000

        # Get CIFAR10 test data
        X_train, Y_train, fn_train, X_test, Y_test, fn_test = data_cifar10(train_start=train_start,
                                                                           train_end=train_end,
                                                                           test_start=test_start,
                                                                           test_end=test_end,
                                                                           preprocess=preprocess_image)
    elif 'svhn' == datasets:
        # choose the method of preprocess image
        preprocess_image = preprocess_image_1

        train_start = 0
        train_end = 73257
        test_start = 0
        test_end = 26032

        # Get SVHN test data
        X_train, Y_train, X_test, Y_test = data_svhn(train_start=train_start,
                                                     train_end=train_end,
                                                     test_start=test_start,
                                                     test_end=test_end,
                                                     preprocess=preprocess_image)

    sess, preds, x, y, model, feed_dict = model_load(datasets, model_path + datasets)

    # Generate random matution matrix for mutations
    store_path = store_path + attack + '/' + datasets + '/' + str(level)
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    result = ''

    sample_path = sample_path + attack + '/' + datasets
    [image_list, image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(sample_path)
    index = np.random.choice(len(image_files), test_num, replace=False)
    image_list = np.asarray(image_list)[index]
    image_files = np.asarray(image_files)[index].tolist()
    predicted_labels = np.asarray(predicted_labels)[index].tolist()

    seed_number = len(image_list)
    if datasets == 'mnist':
        img_rows = 28
        img_cols = 28
        mutation_test = MutationTest(img_rows, img_cols, seed_number, mutation_number, level)
        mutation_test.mutation_generate(mutated, store_path, utils.generate_value_1)
    elif datasets == 'cifar10' or datasets == 'svhn':
        img_rows = 32
        img_cols = 32
        mutation_test = MutationTest(img_rows, img_cols, seed_number, mutation_number, level)
        mutation_test.mutation_generate(mutated, store_path, utils.generate_value_3)

    store_string, result = mutation_test.mutation_test_adv(preprocess_image_1, result,image_list, predicted_labels, sess, x, preds, image_files, feed_dict)

    with open(store_path + "/adv_result.csv", "w") as f:
        f.write(store_string)

    path = store_path + '/ori_jsma'
    if not os.path.exists(path):
        os.makedirs(path)

    preds_test = np.asarray([])
    for i in range(40):
        preds_test = np.concatenate(
            (preds_test, model_argmax(sess, x, preds, X_test[i * 250:(i + 1) * 250], feed=feed_dict)))
    inds_correct = np.asarray(np.where(preds_test == Y_test.argmax(axis=1))[0])
    inds_correct = inds_correct[np.random.choice(len(inds_correct), test_num, replace=False)]
    image_list = X_test[inds_correct]
    real_labels = Y_test[inds_correct].argmax(axis=1)

    np.save(path + '/ori_x.npy', np.asarray(image_list))
    np.save(path + '/ori_y.npy', np.asarray(real_labels))

    image_list = np.load(path + '/ori_x.npy')
    real_labels = np.load(path + '/ori_y.npy')

    store_string, result = mutation_test.mutation_test_ori(result, image_list, sess, x, preds, feed_dict)

    with open(store_path + "/ori_result.csv", "w") as f:
        f.write(store_string)

    with open(store_path + "/result.csv", "w") as f:
        f.write(result)

    # Close TF session
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
    flags.DEFINE_string('attack', 'jsma', 'The type of generating adversaries')
    flags.DEFINE_boolean('mutated', False, 'The mutation list is already generate.')  # default:False
    flags.DEFINE_string('model_path', '../models/das_cnn/','Path to save or load the model file')
    flags.DEFINE_string('store_path', '../results/', 'The path to store results.')
    flags.DEFINE_string('sample_path', '../datasets/adversary/', 'The path to load adversaries.')
    flags.DEFINE_integer('level', 1, 'the level of random mutation region.')
    flags.DEFINE_integer('test_num', 100, 'Number of mutation test targets')
    flags.DEFINE_integer('mutation_num', 500, 'Number of mutation tests')

    tf.app.run()
