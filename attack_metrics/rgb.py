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
from nmutant_util.utils_file import get_data_file, get_data_file_with_Gaussian
from nmutant_util.utils_imgproc import preprocess_image_1

FLAGS = flags.FLAGS

ua = ["fgsm", "blackbox"]
ta = ["jsma", 'cw']

def rgb(datasets, model, samples_path, radius, epoch=49):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    # Object used to keep track of (and return) key accuracies
    tf.reset_default_graph()
    sess, preds, x, y, model, feed_dict = model_load(datasets, model, epoch=epoch)

    [image_list, image_files, real_labels, predicted_labels] = get_data_file_with_Gaussian(datasets, samples_path, radius)

    samples = np.asarray([preprocess_image_1(image.astype('float64')) for image in image_list])

    RGB_UA=0
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
            if np.argmax(probabilities[j])!=real_labels[start+j]:
                RGB_UA+=1

    result = RGB_UA / len(image_list)
    print('Robustness to Gaussian Blur   %.4f' %(result))

    # Close TF session
    sess.close()

    return result


def main(argv=None):
    rgb(datasets = FLAGS.datasets,
         model=FLAGS.model,
         samples_path=FLAGS.samples,
         radius=FLAGS.radius)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model', 'lenet1', 'The name of model')
    flags.DEFINE_string('samples', '../adv_result/mnist/lenet1_fgsm_test', 'The path to load samples.')
    flags.DEFINE_string('radius', '1', 'The Gaussion radius.')
    tf.app.run()
