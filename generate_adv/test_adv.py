import sys
sys.path.append('../')

from nmutant_attack.fgsm import fgsm
from nmutant_attack.cw import cw
from nmutant_attack.jsma import jsma
from nmutant_data.mnist import data_mnist
import tensorflow as tf
from nmutant_data.data import get_shape
import numpy as np
from nmutant_util.utils_file import get_data_file
from nmutant_model.model_operation import model_load, model_eval
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def test_adv(datasets, model_name, samples_path, de=True, attack='fgsm', epoch=9):
    [image_list, _, real_labels, _] = get_data_file(samples_path)
    tf.reset_default_graph()
    sess, preds, x, y, model, feed_dict = model_load(datasets=datasets, model_name=model_name, de=de, attack=attack,
                                                     epoch=epoch)

    def y_one_hot(label):
        y = np.zeros(10)
        y[label] = 1
        return y

    eval_params = {'batch_size': 256}

    labels_adv = np.asarray([y_one_hot(int(label)) for label in real_labels])

    accuracy = model_eval(sess,x,y,preds,np.asarray(image_list), labels_adv, feed_dict, eval_params)
    print(accuracy)

def main(argv=None):
    samples_path = '../adv_result/' + FLAGS.datasets + '/' + FLAGS.attack + '/' + FLAGS.model + '/test_data'
    test_adv(datasets = FLAGS.datasets,
                           model_name=FLAGS.model,
                           samples_path=samples_path,
                           attack=FLAGS.attack,
             epoch=FLAGS.epoch)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model')
    # flags.DEFINE_string('samples', 'test', 'The path to load samples.')  # '../mt_result/mnist_jsma/adv_jsma'
    flags.DEFINE_integer('epoch', 99, '')
    flags.DEFINE_string('attack', 'fgsm', '')

    tf.app.run()
