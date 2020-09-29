from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_util.utils_file import get_data_file
from nmutant_data.data import get_data
from nmutant_util.utils_imgproc import deprocess_image_1

FLAGS = flags.FLAGS

def psd(datasets, model, samples_path, n):
    """
    :param datasets
    :param model
    :param samples_path
    :param n
    :return:
    """
    con = 5
    X_train, Y_train, X_test, Y_test = get_data(datasets)
    X_test = [deprocess_image_1(np.asarray([image])) for image in X_test]

    [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)
    # if datasets=='cifar10':
    #     image_list = [(img*255).reshape(img.shape[0], img.shape[1], img.shape[2]) for img in image_list]
    # else:
    #     image_list = [(img*255).reshape(img.shape[0], img.shape[1]) for img in image_list]
    if datasets == 'cifar10':
        image_list = [deprocess_image_1(np.asarray([img])).reshape(img.shape[0], img.shape[1], img.shape[2]) for img in
                      image_list]
    else:
        image_list = [deprocess_image_1(np.asarray([img])).reshape(img.shape[0], img.shape[1]) for img in image_list]
    
    result = 0.0
    for i in range(len(image_list)):
        index = int(image_files[i].split('_')[-4])
        adv = np.asarray(image_list[i])
        ori = np.asarray(X_test[index])
        if datasets=='cifar10':
            ori=ori.reshape(ori.shape[0], ori.shape[1], ori.shape[2])
        else:
            ori=ori.reshape(ori.shape[1], ori.shape[2])

        #result = result + distance(adv, ori, (n - 1) / 2)
        result = result + distance(adv, ori, n, con)

    print('average perturbation sensitivity distance is %.4f' % (result/len(image_list)))
    #print('average perturbation sensitivity distance is %.4f' % (result/100))
    return result/len(image_list)


def distance(adv, ori, n, constant):
    # change to gray pic
    if 3 == len(adv.shape):
        adv = (adv * np.asarray([0.3, 0.59, 0.11])).astype(np.int32)
        adv = adv.sum(axis=2)
        ori = (ori * np.asarray([0.3, 0.59, 0.11])).astype(np.int32)
        ori = ori.sum(axis=2)
        #print(adv)
    else:
        ori = ori.reshape(ori.shape[0], ori.shape[1])
    

    delta = abs(adv - ori)
    #sd = np.asarray([[0.0] * ori.shape[1]] * ori.shape[0])
    result = 0.0

    for i in range(0, ori.shape[0]):
        for j in range(0, ori.shape[1]):
            if i - n < 0:
                top = 0
            else:
                top = i - n
            if i + n > ori.shape[0]:
                down = ori.shape[0]
            else:
                down = i + n
            if j - n < 0:
                left = 0
            else:
                left = j - n
            if j + n > ori.shape[1]:
                right = ori.shape[1]
            else:
                right = j + n
            region = ori[int(top):int(down+1),int(left):int(right+1)].reshape(-1)
            print(np.asarray(region).shape)

            average = np.mean(region)
            sq = 0.0
            sen = 0.0
            for l in range(0,len(region)):
                sq += math.pow(region[l]-average, 2)
            sd = math.pow(sq / (len(region)), 0.5)
            if sd == 0.0:
                sen = constant  #a constant
            else:
                sen=1/sd

            result+=sen*delta[i][j]
    print(result)
    return result

def main(argv=None):
    psd(datasets = FLAGS.datasets,
        model=FLAGS.model,
        samples_path=FLAGS.samples,
        n=FLAGS.n)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model')
    flags.DEFINE_string('samples', '../adv_result/mnist/fgsm/lenet1/0.2', 'The path to load samples.')
    flags.DEFINE_integer('n', '1', 'region size')

    tf.app.run()
