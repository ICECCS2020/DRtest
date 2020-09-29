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

def ald(datasets, model, samples_path, p, epoch=49):
    """
    :param datasets
    :param model
    :param samples_path
    :param p
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
    
    distortion = 0
    for i in range(len(image_list)):
        index = int(image_files[i].split('_')[-4])
        distortion = distortion + distortion_measure(np.asarray(image_list[i]), np.asarray(X_test[index]), p)

    result = distortion / len(image_list)
    print('average L-%s distortion is %.4f' % (p, result))

    return result

def distortion_measure(adv, ori, p):
    # change to gray pic
    if 3 == len(adv.shape):
        adv = adv * np.asarray([0.3 , 0.59, 0.11])
        adv = adv.sum(axis=2)
        ori = ori * np.asarray([0.3 , 0.59, 0.11])
        ori = ori.sum(axis=2)
    adv = adv.reshape(-1)
    ori = ori.reshape(-1)

    distance = 0
    ori_distance = 0
    if '0' == p:
        distance = distance + sum([int(i) for i in adv != ori])
        ori_distance = ori_distance + len(adv)
    elif '1' == p:
        distance = distance + sum(abs(adv - ori))
        ori_distance = ori_distance + sum(ori)
    elif '2' == p:
        distance = distance + sum((adv - ori) ** 2) ** 0.5
        ori_distance = ori_distance + sum(ori ** 2) ** 0.5
    elif 'inf' == p:
        distance = distance + max(abs(adv - ori))
        ori_distance = ori_distance + max(ori)

    return 1.0 * distance / ori_distance

def main(argv=None):
    ald(datasets = FLAGS.datasets,
        model=FLAGS.model,
        samples_path=FLAGS.samples,
        p=FLAGS.p)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model')
    flags.DEFINE_string('samples', '../adv_result/mnist/fgsm/lenet1/0.1', 'The path to load samples.')
    flags.DEFINE_string('p', '0', 'p-norm distance,p=0,1,2,inf')

    tf.app.run()
