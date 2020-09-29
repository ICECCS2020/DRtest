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
from nmutant_util.utils_tf import model_train, model_eval, batch_eval
import scipy
import scipy.stats
from nmutant_util.utils_tf import model_prediction

FLAGS = flags.FLAGS

def cos(datasets, model, de_model, attack='fgsm', epoch=49, de_epoch=49):

    tf.reset_default_graph()
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    # Object used to keep track of (and return) key accuracies

    print("load defense model.")
    sess, preds, x, y, model, feed_dict = model_load(datasets, model, epoch=epoch)

    X_train, Y_train, X_test, Y_test = get_data(datasets)
    input_shape, nb_classes = get_shape(datasets)
    feed_dict_de = None
    #result_nor=sess.run(preds, feed_dict={x:X_test})
    result_nor = model_prediction(sess, x, preds, X_test, feed=feed_dict, datasets=datasets)
    #print(result_nor)
    #print(model)
    #print(get_model_dict())


    
    tf.reset_default_graph()
    sess, preds_de, x, y, model_de, feed_dict = model_load(datasets, de_model, de=True, attack=attack, epoch=de_epoch)
    #result_de=sess.run(preds_de, feed_dict={x:X_test})
    result_de = model_prediction(sess, x, preds_de, X_test, feed=feed_dict, datasets=datasets)
    #print(result_de)
    # print('average confidence of adversarial class %.4f' %(result))
    result=0
    num=0
    js=0.
    for i in range(Y_test.shape[0]):
        if (np.argmax(Y_test[i])==np.argmax(result_nor[i])) and (np.argmax(Y_test[i])==np.argmax(result_de[i])):
            num+=1
            p=result_nor[i]
            q=result_de[i]
            M=(p+q)/2
            js=js+0.5*scipy.stats.entropy(p, M)+0.5*scipy.stats.entropy(q, M)
    # Close TF session
    result=js/num
    print("JS divergence: ", result)
    sess.close()

    return result


def main(argv=None):
    cos(datasets = FLAGS.datasets,
        model=FLAGS.model,
        de_model=FLAGS.de_model)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet5', 'The name of model')
    flags.DEFINE_string('de_model', 'lenet5', 'The name of defense-enhanced model.')

    tf.app.run()
