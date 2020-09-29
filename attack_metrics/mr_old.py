from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_model.model_operation import model_load
from nmutant_util.utils_file import get_data_file
from nmutant_data.data import get_data, get_shape
from nmutant_util.utils_tf import model_argmax

FLAGS = flags.FLAGS

ua = ["fgsm", "blackbox"]
ta = ["jsma", 'cw']

def mr(datasets, model, samples_path):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    tf.reset_default_graph()
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    input_shape, nb_classes = get_shape(datasets)
    sess, preds, x, y, model, feed_dict = model_load(datasets, model)

    preds_test = np.asarray([])
    n_batches = int(np.ceil(1.0 * X_test.shape[0] / 256))
    for i in range(n_batches):
        start = i * 256
        end = np.minimum(len(X_test), (i + 1) * 256)
        preds_test = np.concatenate(
            (preds_test, model_argmax(sess, x, preds, X_test[start:end], feed=feed_dict)))
    inds_correct = np.asarray(np.where(preds_test != Y_test.argmax(axis=1))[0])
    X_test = X_test[inds_correct]

    [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
    for a in ua:
        if a in samples_path:
            result = len(image_files) / len(X_test)
            print('misclassification ratio is %.4f' % (result))
            return result

    for a in ta:
        if a in samples_path:
            result = len(image_files) / (len(X_test) * (nb_classes - 1))
            print('misclassification ratio is %.4f' % (result))
            return result

def main(argv=None):
    mr(datasets = FLAGS.datasets,
       model=FLAGS.model,
       samples_path=FLAGS.samples)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model')
    flags.DEFINE_string('samples', '../mt_result/cifar10_jsma/adv_jsma', 'The path to load samples.')

    tf.app.run()
