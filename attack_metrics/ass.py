from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_util.utils_file import get_data_file
from nmutant_data.data import get_data
from nmutant_util.utils_imgproc import deprocess_image_1

FLAGS = flags.FLAGS

def ass(datasets, model, samples_path):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    tf.reset_default_graph()
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    X_test = [deprocess_image_1(np.asarray([image])) for image in X_test]

    [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
    if datasets=='cifar10':
        image_list = [(img*255).reshape(img.shape[0], img.shape[1], img.shape[2]) for img in image_list]
    else:
        image_list = [(img*255).reshape(img.shape[0], img.shape[1]) for img in image_list]
    result = 0
    for i in range(len(image_list)):
        index = int(image_files[i].split('_')[-4])
        result = result + ssim(np.asarray(image_list[i]), np.asarray(X_test[index]))

    result = result / len(image_list)
    print('average structural similarity is %.4f' % (result))

    return result

def ssim(adv, ori):
    #change to gray pic
    if 3 == len(adv.shape):
        adv = adv * np.asarray([0.3 , 0.59, 0.11])
        adv = adv.sum(axis=2)
        ori = ori * np.asarray([0.3 , 0.59, 0.11])
        ori = ori.sum(axis=2)
    adv = adv.reshape(-1)
    ori = ori.reshape(-1)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    c3 = c2 / 2
    alpha = 1
    beta = 1
    gama = 1

    miu_x = adv.mean()
    miu_y = ori.mean()

    theta_x = adv.std(ddof=1)
    theta_y = ori.std(ddof=1)
    theta_xy = sum((adv - miu_x) * (ori - miu_y)) / (len(adv) - 1)

    l = (2 * miu_x * miu_y + c1) / (miu_x ** 2 + miu_y ** 2 + c1)
    c = (2 * theta_x * theta_y + c2) / (theta_x ** 2 + theta_y ** 2 + c2)
    s = (theta_xy + c3) / (theta_x * theta_y + c3)

    return (l ** alpha) * (c ** beta) * (s ** gama)

def main(argv=None):
    ass(datasets = FLAGS.datasets,
        model=FLAGS.model,
        samples_path=FLAGS.samples)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model')
    flags.DEFINE_string('samples', '../mt_result/cifar10_jsma/adv_jsma', 'The path to load samples.')

    tf.app.run()
