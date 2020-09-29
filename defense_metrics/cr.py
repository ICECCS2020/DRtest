from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_model.model_operation import model_load, get_model_dict
from nmutant_data.data import get_data, get_shape
from nmutant_util.configs import path
from nmutant_util.utils_tf import model_argmax

FLAGS = flags.FLAGS

def correct(datasets, model_name, X_test, Y_test, de=False, attack='fgsm', epoch=49):
    tf.reset_default_graph()
    sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack, epoch=epoch)
    preds_test = np.asarray([])
    n_batches = int(np.ceil(1.0 * X_test.shape[0] / 256))
    for i in range(n_batches):
        start = i * 256
        end = np.minimum(len(X_test), (i + 1) * 256)
        preds_test = np.concatenate(
            (preds_test, model_argmax(sess, x, preds, X_test[start:end], feed=feed_dict)))
    inds_correct = np.asarray(np.where(preds_test == Y_test.argmax(axis=1))[0])
    sess.close()
    tf.reset_default_graph()
    return inds_correct

def cr(datasets, model_name, attack, epoch=49, de_epoch=49):
    """
    :param datasets
    :param model
    :return:
    """
    # Object used to keep track of (and return) key accuracies
    X_train, Y_train, X_test, Y_test = get_data(datasets)

    ori_correct = set(correct(datasets, model_name, X_test, Y_test, epoch=epoch))
    de_correct = set(correct(datasets, model_name, X_test, Y_test, de=True, attack=attack, epoch=de_epoch))

    full = set([i for i in range(len(X_test))])

    ori_wrong = full - ori_correct
    de_wrong = full - de_correct

    crr = 1.0 * len(ori_wrong & de_correct) / len(X_test)
    csr = 1.0 * len(ori_correct & de_wrong) / len(X_test)
    cav = 1.0 * (len(de_correct) - len(ori_correct)) / len(X_test)

    print('classification rectify ratio is %.4f' %(crr))
    print('classification sacrifice ratio is %.4f' % (csr))
    print('classification accuracy variance is %.4f' %(cav))

    return crr, csr, cav


def main(argv=None):
    # cr(datasets = FLAGS.datasets,
    #     model_name=FLAGS.model, attack=FLAGS.attack, epoch=FLAGS.epoch, de_epoch=FLAGS.de_epoch)
    dict = {"vgg11":64, "vgg13":49, "vgg16":99, "vgg19":99, "resnet18":94, "resnet34":42, "resnet50":97,"resnet101":484,"resnet152":403,
            "googlenet12":75,"googlenet16":87,"googlenet22":54}
    for model in dict:
        epoch = dict[model]
        for attack in ["fgsm", "cw", "jsma"]:
            f = open('cifar_defense.txt', 'a')
            crr, csr, cav = cr(datasets=FLAGS.datasets,
                model_name=model, attack=attack, epoch=epoch, de_epoch=99)
            f.write('\n')
            f.write('datasets: '+FLAGS.datasets+' model:' + model + ' attack:'+attack+ ':')
            f.write('\n')
            f.write('crr:' +str(crr))
            f.write('\n')
            f.write('csr:' + str(csr))
            f.write('\n')
            f.write('cav:' + str(cav))
            f.write('\n')
            f.close()

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('model', 'vgg11', 'The name of model')
    flags.DEFINE_string('attack', 'fgsm', 'attack')
    flags.DEFINE_integer('epoch', 64, '')
    flags.DEFINE_integer('de_epoch', 99, '')
    tf.app.run()
