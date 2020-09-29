from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import os
from PIL import Image, ImageFilter

sys.path.append("../")

from nmutant_model.model_operation import model_load
from nmutant_util.utils_file import get_data_file
from nmutant_util.utils_imgproc import preprocess_image_1,deprocess_image_1
from nmutant_util.utils_tf import model_argmax
from scipy.misc import imsave, imread

FLAGS = flags.FLAGS

def ric(datasets, model, samples_path, quality, epoch=49):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    # Object used to keep track of (and return) key accuracies
    sess, preds, x, y, model, feed_dict = model_load(datasets, model, epoch=epoch)

    [image_list, image_files, real_labels, predicted_labels] = get_data_file(samples_path)

    ori_path = samples_path.replace('test_data','ric_ori')
    if not os.path.exists(ori_path):
        os.makedirs(ori_path)
    ic_path = samples_path.replace('test_data', 'ric_ic')
    if not os.path.exists(ic_path):
        os.makedirs(ic_path)

    count = 0
    for i in range(len(image_list)):
        #jj=np.asarray(image_list[i:i+1])
        #print(jj.shape)
        if datasets == 'mnist':
            adv_img_deprocessed = deprocess_image_1(np.asarray(image_list[i:i+1]))[0]
        elif datasets == 'cifar10':
            adv_img_deprocessed = deprocess_image_1(np.asarray(image_list[i:i+1]))

        saved_adv_image_path = os.path.join(ori_path, image_files[i].replace("npy","png"))
        imsave(saved_adv_image_path, adv_img_deprocessed)

        output_IC_path = os.path.join(ic_path, image_files[i].replace("npy","jpg"))

        cmd = '../../guetzli/bin/Release/guetzli --quality {} {} {}'.format(quality, saved_adv_image_path, output_IC_path)
        assert os.system(cmd) == 0, 'guetzli tool should be install before, https://github.com/google/guetzli'

        if datasets == 'cifar10':
            IC_image = Image.open(output_IC_path).convert('RGB')
            IC_image = np.asarray([np.array(IC_image).astype('float32') / 255.0])
            #IC_image=IC_image.reshape(32, 32, 3)
        elif datasets == 'mnist':
            IC_image = Image.open(output_IC_path).convert('L')
            IC_image = np.expand_dims(np.array(IC_image).astype('float32'), axis=0) / 255.0
            IC_image = IC_image.reshape(-1,28,28,1)

        if model_argmax(sess, x, preds, IC_image, feed=feed_dict) != int(real_labels[i]):
            count = count + 1

    result = 1.0 * count / len(image_list)
    print('Robustness to image compression is %.4f' %(result))

    # Close TF session
    sess.close()

    return result


def main(argv=None):
    ric(datasets = FLAGS.datasets,
         model=FLAGS.model,
         samples_path=FLAGS.samples,
        quality = FLAGS.quality,
    epoch = FLAGS.epoch)

if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'cifar10', 'The target datasets.')
    flags.DEFINE_string('model', 'vgg11', 'The name of model')
    flags.DEFINE_string('samples', '/mnt/dyz/adv_result/cifar10/fgsm/vgg11/test_data', 'The path to load samples.')
    flags.DEFINE_integer('quality', 90, '')
    flags.DEFINE_integer('epoch',49,'')
    tf.app.run()