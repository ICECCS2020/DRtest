from __future__ import print_function

import math
import random

import numpy as np

from nmutant_util.utils_tf import model_argmax
from input_mutation.utils import c_occl, c_light, c_black


class MutationTest:

    '''
        Mutation testing for the training dataset
        :param img_rows:
        :param img_cols:
        :param seed_number:
        :param mutation_number:
    '''

    img_rows = 28
    img_cols = 28
    seed_number = 500
    mutation_number = 1000
    mutations = []
    level = 1

    def __init__(self, img_rows, img_cols, seed_number, mutation_number, level):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.seed_number = seed_number
        self.mutation_number = mutation_number
        self.level = level

    def mutation_matrix(self, generate_value):

        method = random.randint(1, 3)
        trans_matrix = generate_value(self.img_rows, self.img_cols)
        rect_shape = (random.randint(1, 3), random.randint(1, 3))
        start_point = (
            random.randint(0, self.img_rows - rect_shape[0]),
            random.randint(0, self.img_cols - rect_shape[1]))

        if method == 1:
            transformation = c_light(trans_matrix)
        elif method == 2:
            transformation = c_occl(trans_matrix, start_point, rect_shape)
        elif method == 3:
            transformation = c_black(trans_matrix, start_point, rect_shape)

        return np.asarray(transformation[0])
        # trans_matrix = generate_value(self.img_rows, self.img_cols, self.level)
        # return np.asarray(trans_matrix)

    def mutation_generate(self, mutated, path, generate_value):
        if mutated:
            self.mutations = np.load(path + "/mutation_list.npy")
        else:
            for i in range(self.mutation_number):
                mutation = self.mutation_matrix(generate_value)
                self.mutations.append(mutation)
            np.save(path + "/mutation_list.npy", self.mutations)

    def mutation_test_adv(self, preprocess_image, result,image_list, predicted_labels, sess, x, preds, image_files=None, feed_dict=None, mutate=False):
        store_string = ''
        mutated = False

        label_change_numbers = []

        # Iterate over all the test data
        # count = 0
        for i in range(len(image_list)):
            # if count % 100 == 0 and mutate:
            #     path = '../mt_result/cifar10_test/mutation/'
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     path = path + image_files[i].split('_')[0] + '_' + str(count)
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     self.mutation_generate(mutated, path, utils.generate_value_3)
            # count += 1
            ori_img = preprocess_image(image_list[i].astype('float64'))
            # ori_img = image_list[i]
            orig_label = predicted_labels[i]

            # pxzhang
            feed = {x: np.expand_dims(ori_img.copy(), 0)}
            if feed_dict is not None:
                feed.update(feed_dict)
            probabilities = sess.run(preds, feed)[0]
            max_p = max(probabilities)
            p = np.argmax(probabilities)
            # new = utils.input_preprocessing(preds, x, 0.001, 0.0, 1.0)
            # ori_img = sess.run(new, feed_dict={x: np.expand_dims(ori_img.copy(), 0)})[0]

            label_changes = 0

            imgs = np.asarray([ori_img.tolist()] * self.mutation_number)
            mu_imgs = imgs + np.asarray(self.mutations) * self.level
            n_batches = int(np.ceil(1.0 * mu_imgs.shape[0] / 256))
            mu_labels = []
            for j in range(n_batches):
                start = j * 256
                end = np.minimum(len(mu_imgs), (j + 1) * 256)
                mu_labels = mu_labels + model_argmax(sess, x, preds, mu_imgs[start:end], feed=feed_dict).tolist()
            for mu_label in mu_labels:
                if mu_label != int(orig_label):
                    label_changes += 1

            # for j in range(self.mutation_number):
            #     img = ori_img.copy()
            #     add_mutation = self.mutations[j]#[0]
            #     mu_img = img + add_mutation
            #
            #     # Predict the label for the mutation
            #     mu_img = np.expand_dims(mu_img, 0)
            #
            #     mu_label = model_argmax(sess, x, preds, mu_img, feed=feed_dict)
            #
            #     if mu_label != int(orig_label):
            #         label_changes += 1

            label_change_numbers.append(label_changes)
            # pxzhang
            store_string = store_string + image_files[i] + "," + str(p) + "," + str(max_p) + "," + str(label_changes) + "\n"

        label_change_numbers = np.asarray(label_change_numbers)
        adv_average = round(np.mean(label_change_numbers), 2)
        adv_std = np.std(label_change_numbers)
        adv_99ci = round(2.576 * adv_std / math.sqrt(len(label_change_numbers)), 2)
        if image_files == None:
            result = result + 'adv,' + ',' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(adv_99ci) + '\n'
        else:
            result = result + 'adv_' + image_files[0].split('_')[0] + ',' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(adv_99ci) + '\n'

        return store_string, result

    def mutation_test_ori(self, result, image_list, sess, x, preds, feed_dict=None):
        store_string = ''

        label_change_numbers = []
        # Iterate over all the test data
        for i in range(len(image_list)):
            ori_img = image_list[i]
            orig_label = model_argmax(sess, x, preds, np.asarray([image_list[i]]))

            # pxzhang
            feed = {x:np.expand_dims(ori_img.copy(),0)}
            if feed_dict is not None:
                feed.update(feed_dict)
            probabilities = sess.run(preds, feed)[0]
            max_p = max(probabilities)
            # new = utils.input_preprocessing(preds, x, 0.001, 0.0, 1.0)
            # ori_img = sess.run(new, feed_dict={x: np.expand_dims(ori_img.copy(), 0)})[0]

            label_changes = 0

            imgs = np.asarray([ori_img.tolist()] * self.mutation_number)
            mu_imgs = imgs + np.asarray(self.mutations) * self.level
            n_batches = int(np.ceil(1.0 * mu_imgs.shape[0] / 256))
            mu_labels = []
            for j in range(n_batches):
                start = j * 256
                end = np.minimum(len(mu_imgs), (j + 1) * 256)
                mu_labels = mu_labels + model_argmax(sess, x, preds, mu_imgs[start:end], feed=feed_dict).tolist()
            for mu_label in mu_labels:
                if mu_label != int(orig_label):
                    label_changes += 1
            # for j in range(self.mutation_number):
            #     img = ori_img.copy()
            #     add_mutation = self.mutations[j][0]
            #     mu_img = img + add_mutation
            #
            #     # Predict the label for the mutation
            #     mu_img = np.expand_dims(mu_img, 0)
            #
            #     mu_label = model_argmax(sess, x, preds, mu_img, feed=feed_dict)
            #
            #     if mu_label != int(orig_label):
            #         label_changes += 1

            label_change_numbers.append(label_changes)
            # pxzhang
            store_string = store_string + str(i) + "," + str(orig_label) + "," + str(max_p) + "," + str(label_changes) + "\n"

        label_change_numbers = np.asarray(label_change_numbers)
        adv_average = round(np.mean(label_change_numbers), 2)
        adv_std = np.std(label_change_numbers)
        adv_99ci = round(2.576 * adv_std / math.sqrt(len(label_change_numbers)), 2)
        result = result + 'ori,' + str(adv_average) + ',' + str(round(adv_std, 2)) + ',' + str(adv_99ci) + '\n'

        return store_string, result

