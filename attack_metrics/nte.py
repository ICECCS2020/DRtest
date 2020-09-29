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
from nmutant_util.utils_imgproc import preprocess_image_1

FLAGS = flags.FLAGS

def nte(datasets, model, samples_path, epoch=49):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    tf.reset_default_graph()
    # Object used to keep track of (and return) key accuracies
    sess, preds, x, y, model, feed_dict = model_load(datasets, model, epoch=epoch)

    [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)

    samples = np.asarray([preprocess_image_1(image.astype('float64')) for image in image_list])
    samples = np.asarray(image_list)
    pbs = []
    n_batches = int(np.ceil(samples.shape[0] / 256))
    for i in range(n_batches):
        start = i * 256
        end = np.minimum(len(samples), (i + 1) * 256)
        feed = {x: samples[start:end]}
        if feed_dict is not None:
            feed.update(feed_dict)
        probabilities = sess.run(preds, feed)
        #print(probabilities[1])
        for j in range(len(probabilities)):
            pro_adv_max=probabilities[j][predicted_labels[start+j]]
            temp=np.delete(probabilities[j], predicted_labels[start+j], axis=0)
            pro_adv_top2=np.max(temp)
            pbs.append(pro_adv_max-pro_adv_top2)
    result = sum(pbs) / len(pbs)
    print('Noise Tolerance Estimation  %.4f' %(result))

    # Close TF session
    sess.close()

    return result


def main(argv=None):
    nte(datasets = FLAGS.datasets,
         model=FLAGS.model,
         samples_path=FLAGS.samples)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet5', 'The name of model')
    flags.DEFINE_string('samples', '../adv_result/fgsm/mnist/adv', 'The path to load samples.')

    tf.app.run()
