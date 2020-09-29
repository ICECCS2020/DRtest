import sys

import tensorflow as tf
from tensorflow.python.platform import flags

sys.path.append("../")

from nmutant_detection import utils
import os
from nmutant_detection.detection import detector
from nmutant_model.model_operation import model_load
from nmutant_util.utils_imgproc import preprocess_image_1

FLAGS = flags.FLAGS

def directory_detect(datasets, dir_path, normal, store_path, ad, sess, preds, x, feed_dict):

    print('--- Extracting images from: ', dir_path)
    if normal:
        [adv_image_list, adv_image_files, real_labels, predicted_labels] = utils.get_normal_data_mutation_test(dir_path)
    else:
        [adv_image_list, adv_image_files, real_labels, predicted_labels] = utils.get_data_mutation_test(dir_path)
    adv_count = 0
    not_decided_images = 0
    total_mutation_counts = []
    label_change_mutation_counts = []
    suc_total_mutation_counts = []
    suc_label_change_mutation_counts = []

    print('--- Evaluating inputs ---')

    if not os.path.exists(store_path):
        os.makedirs(store_path)
    detector_results = []
    summary_results = []
    for i in range(len(adv_image_list)):
        # # print('- Running image ', i)
        ori_img = preprocess_image_1(adv_image_list[i].astype('float32'))

        orig_label = predicted_labels[i]
        [result, decided, total_mutation_count, label_change_mutation_count] = ad.detect(ori_img, orig_label, sess, x,
                                                                                         preds, feed_dict)

        detector_results.append(adv_image_files[i] + ',' + str(result) + ',' + str(decided) + ',' + str(total_mutation_count) + ',' + str(label_change_mutation_count))

        if result:
            adv_count += 1
            if not normal: # Record the counts for adversaries
                suc_total_mutation_counts.append(total_mutation_count)
                suc_label_change_mutation_counts.append(label_change_mutation_count)

        if normal and not result: # Record the counts for normals
            suc_total_mutation_counts.append(total_mutation_count)
            suc_label_change_mutation_counts.append(label_change_mutation_count)

        if not decided:
            not_decided_images += 1

        total_mutation_counts.append(total_mutation_count)
        label_change_mutation_counts.append(label_change_mutation_count)

    with open(store_path + "/detection_result.csv", "w") as f:
        for item in detector_results:
            f.write("%s\n" % item)

    summary_results.append('adv_num,' + str(len(adv_image_list)))
    summary_results.append('identified_num,' + str(adv_count))
    summary_results.append('undecided_num,' + str(not_decided_images))

    if normal:
        summary_results.append('accuracy,' + str(1 - float(adv_count)/len(total_mutation_counts)))
    else:
        summary_results.append('accuracy,' + str(float(adv_count)/len(total_mutation_counts)))

    if len(suc_label_change_mutation_counts) > 0 and not normal:
        summary_results.append(
            'avg_mutation_num,' + str(sum(suc_total_mutation_counts) / len(suc_total_mutation_counts)))
        summary_results.append(
            'avg_lc_num,' + str(float(sum(suc_label_change_mutation_counts)) / len(suc_label_change_mutation_counts)))

    if len(suc_label_change_mutation_counts) > 0 and normal:
        summary_results.append(
            'avg_mutation_num,' + str(sum(suc_total_mutation_counts) / len(suc_total_mutation_counts)))
        summary_results.append(
            'avg_lc_num,' + str(float(sum(suc_label_change_mutation_counts)) / len(suc_label_change_mutation_counts)))

    summary_results.append(total_mutation_counts)
    summary_results.append(label_change_mutation_counts)

    with open(store_path + "/detection_summary_result.csv", "w") as f:
        for item in summary_results:
            f.write("%s\n" % item)

    print('- Total adversary images evaluated: ', len(adv_image_list))
    print('- Identified adversaries: ', adv_count)
    print('- Not decided images: ', not_decided_images)
    if len(suc_label_change_mutation_counts) > 0:
        print('- Average mutation needed: ', sum(suc_total_mutation_counts) / len(suc_total_mutation_counts))
        print('- Average label change mutations: ',
              float(sum(suc_label_change_mutation_counts)) / len(suc_label_change_mutation_counts))
    else:
        summary_results.append(
            'avg_mutation_num,' + str(sum(total_mutation_counts) / len(total_mutation_counts)))
        summary_results.append(
            'avg_lc_num,' + str(float(sum(label_change_mutation_counts)) / len(label_change_mutation_counts)))

def main(argv=None):
    datasets = FLAGS.datasets
    attack_type = FLAGS.attack_type

    # detector config
    k_nor = FLAGS.k_nor
    mu = FLAGS.mu
    level = FLAGS.level
    max_mutations = FLAGS.max_iteration
    normal = False

    indifference_region_ratio = mu - 1
    alpha = 0.05
    beta = 0.05
    if 'mnist' == datasets:
        rgb = False
        image_rows = 28
        image_cols = 28
    elif 'cifar10' == datasets:
        rgb = True
        image_rows = 32
        image_cols = 32

    print('--- Dataset: ', datasets, 'attack type: ', attack_type)

    sess, preds, x, y, model, feed_dict = model_load(datasets, FLAGS.model_name, FLAGS.epoch)

    adv_image_dir = FLAGS.sample_path + '/' + attack_type + '/test'
    if attack_type.__eq__('normal'):
        normal = True

    store_path = FLAGS.store_path + datasets + '_' + attack_type + '/level=' + str(level)+',mm=' + \
                     str(max_mutations) + '/mu=' + str(mu) + ',irr=' + str(indifference_region_ratio)

    # Detection
    ad = detector(k_nor, mu, image_rows, image_cols, level, rgb, max_mutations, alpha, beta, k_nor*indifference_region_ratio)
    print('--- Detector config: ', ad.print_config())
    directory_detect(datasets, adv_image_dir, normal, store_path, ad, sess, preds, x, feed_dict)

if __name__ == '__main__':
    # flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    # flags.DEFINE_string('model_path', '../models/integration/mnist', 'The path to load model.')
    # flags.DEFINE_string('sample_path', '../datasets/experiment/mnist', 'The path storing samples.')
    flags.DEFINE_string('datasets', 'mnist', 'The target datasets.')
    flags.DEFINE_string('model_name', 'lenet5', 'The path to load model.')
    flags.DEFINE_integer('epoch', 9, 'The epoch of model for use.')
    flags.DEFINE_string('sample_path', '../datasets/experiment/mnist', 'The path storing samples.')
    flags.DEFINE_string('store_path', '../detection/', 'The path to store result.')
    flags.DEFINE_string('attack_type', 'fgsm', 'attack_type')
    flags.DEFINE_float('k_nor', 0.0017, 'normal ratio change')
    flags.DEFINE_float('mu', 1.2, 'mu parameter of the detection algorithm')
    flags.DEFINE_integer('level', 1, 'the level of random mutation region.')
    flags.DEFINE_integer('max_iteration', 2000, 'max iteration of mutation')

    tf.app.run()
