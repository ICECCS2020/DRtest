from __future__ import division, absolute_import, print_function

import os
import multiprocessing as mp
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, cdist, squareform
from nmutant_util.utils_tf import model_prediction

# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
STDEVS = {
    'mnist': {'fgsm': 0.310, 'bim-a': 0.128, 'bim-b': 0.265},
    'cifar': {'fgsm': 0.050, 'bim-a': 0.009, 'bim-b': 0.039},
    'svhn': {'fgsm': 0.132, 'bim-a': 0.015, 'bim-b': 0.122}
}
# Set random seed
np.random.seed(0)

def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < 0.99)[0]
    assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = 1.

    return np.reshape(x, original_shape)


def get_noisy_samples(X_test, X_test_adv, dataset, attack):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    if attack in ['jsma', 'cw']:
        X_test_noisy = np.zeros_like(X_test)
        for i in range(len(X_test)):
            # Count the number of pixels that are different
            nb_diff = len(np.where(X_test[i] != X_test_adv[i])[0])
            # Randomly flip an equal number of pixels (flip means move to max
            # value of 1)
            X_test_noisy[i] = flip(X_test[i], nb_diff)
    else:
        warnings.warn("Using pre-set Gaussian scale sizes to craft noisy "
                      "samples. If you've altered the eps/eps-iter parameters "
                      "of the attacks used, you'll need to update these. In "
                      "the future, scale sizes will be inferred automatically "
                      "from the adversarial samples.")
        # Add Gaussian noise to the samples
        X_test_noisy = np.minimum(
            np.maximum(
                X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                          size=X_test.shape),
                0
            ),
            1
        )

    return X_test_noisy

def get_mc_predictions(sess, x, preds, samples, nb_iter=50):
    """
    TODO
    :param model:
    :param X:
    :param nb_iter:
    :param batch_size:
    :return:
    """
    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(model_prediction(sess, x, preds, samples))

    return np.asarray(preds_mc)

# def get_deep_representations(sess, x, input_data, model, feed_dict):
#     """
#     TODO
#     :param model:
#     :param X:
#     :param batch_size:
#     :return:
#     """
#     # last hidden layer is always at index -4
#     temp = x
#     for i in range(len(model.layers)):
#         layer = model.layers[i]
#         temp = layer.fprop(temp)
#         # if 'Flatten' not in layer.name and 'Input' not in layer.name:
#         if i == len(model.layers) - 4:
#             feed= {x: input_data}
#             if feed_dict is not None:
#                 feed.update(feed_dict)
#             output = sess.run(temp, feed_dict = feed)
#             break
#     return output.reshape(len(input_data), -1)

def get_deep_representations(sess, x, input_data, model, feed_dict):
    """
    TODO
    :param model:
    :param X:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index -4
    temp = x
    output = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        temp = layer.fprop(temp)
        # if 'Flatten' not in layer.name and 'Input' not in layer.name:
        if i == len(model.layers) - 4:
            n_batches = int(np.ceil(1.0 * input_data.shape[0] / 256))
            for j in range(n_batches):
                start = j * 256
                end = np.minimum(len(input_data), (j + 1) * 256)
                feed = {x: input_data[start:end]}
                if feed_dict is not None:
                    feed.update(feed_dict)
                r = sess.run(temp, feed_dict = feed)
                output = output + r.tolist()
            break
    output = np.asarray(output)
    return output.reshape(len(input_data), -1)

def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]

def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results

def normalize(normal, adv):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv)))

    return total[:n_samples], total[n_samples:]

def features(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    return values, labels
    # lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr

def train_lr(X, y):
    """
    TODO
    :param X: the data samples
    :param y: the labels
    :return:
    """
    lr = LogisticRegressionCV(n_jobs=-1).fit(X, y)
    return lr

def compute_roc(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.ones_like(probs_neg), np.zeros_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def compute_roc_lid(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score

def random_split(X, Y):
    """
    Random split the data into 80% for training and 20% for testing
    :param X:
    :param Y:
    :return:
    """
    print("random split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    num_train = int(num_samples * 0.8)
    rand_pert = np.random.permutation(num_samples)
    X = X[rand_pert]
    Y = Y[rand_pert]
    X_train, X_test = X[:num_train], X[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    return X_train, Y_train, X_test, Y_test

def block_split(X, Y, num_train):
    """
    Split the data into 80% for training and 20% for testing
    in a block size of 100.
    :param X:
    :param Y:
    :return:
    """
    print("Isolated split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    partition = int(num_samples / 2)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition:], Y[partition:]

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    # index = np.arange(X.shape[0])
    # np.random.shuffle(index)
    # X = X[index]
    # y = y[index]

    return X, y

# lid of a batch of query points X
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def get_lids_random_batch(sess, x, model, feed_dict, X, X_adv, dataset, k=20, batch_size=100):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model:
    :param X: normal images
    :param X_adv: advserial images
    :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    lid_dim = len(model.layers) + 1
    # print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start

        start_adv = i_batch * adv_batch_size
        end_adv = np.minimum(len(X_adv), (i_batch + 1) * adv_batch_size)
        n_feed_adv = end_adv - start_adv

        lid_batch = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv = np.zeros(shape=(n_feed_adv, lid_dim))

        temp = x
        for i in range(len(model.layers)):
            layer = model.layers[i]
            temp = layer.fprop(temp)
            feed = {x: X[start:end]}
            if feed_dict is not None:
                feed.update(feed_dict)
            X_act = sess.run(temp, feed_dict=feed)
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))

            feed = {x: X_adv[start_adv:end_adv]}
            if feed_dict is not None:
                feed.update(feed_dict)
            X_adv_act = sess.run(temp, feed_dict=feed)
            X_adv_act = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed_adv, -1))

            lid_batch[:, i] = mle_batch(X_act, X_act, k=k)
            lid_batch_adv[:, i] = mle_batch(X_act, X_adv_act, k=k)

        return lid_batch, lid_batch_adv

    lids = []
    lids_adv = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    adv_batch_size = int(np.ceil(len(X_adv) / n_batches))
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)

    lids = np.asarray(lids, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_adv


# def get_lids_random_batch(sess, x, model, feed_dict, X, X_adv, dataset, k=20, batch_size=100):
#     """
#     Get the local intrinsic dimensionality of each Xi in X_adv
#     estimated by k close neighbours in the random batch it lies in.
#     :param model:
#     :param X: normal images
#     :param X_adv: advserial images
#     :param dataset: 'mnist', 'cifar', 'svhn', has different DNN architectures
#     :param k: the number of nearest neighbours for LID estimation
#     :param batch_size: default 100
#     :return: lids: LID of normal images of shape (num_examples, lid_dim)
#             lids_adv: LID of advs images of shape (num_examples, lid_dim)
#     """
#     X_acts = [X]
#     X_adv_acts = [X_adv]
#     temp = x
#     for i in range(len(model.layers)):
#         layer = model.layers[i]
#         temp = layer.fprop(temp)
#         n_batches = int(np.ceil(1.0 * X.shape[0] / 256))
#         layer_output = []
#         for j in range(n_batches):
#             start = j * 256
#             end = np.minimum(len(X), (j + 1) * 256)
#             feed = {x: X[start:end]}
#             if feed_dict is not None:
#                 feed.update(feed_dict)
#             r = sess.run(temp, feed_dict=feed)
#             layer_output = layer_output + r.tolist()
#         X_acts.append(layer_output)
#
#         n_batches = int(np.ceil(1.0 * X_adv.shape[0] / 256))
#         layer_output = []
#         for j in range(n_batches):
#             start = j * 256
#             end = np.minimum(len(X_adv), (j + 1) * 256)
#             feed = {x: X_adv[start:end]}
#             if feed_dict is not None:
#                 feed.update(feed_dict)
#             r = sess.run(temp, feed_dict=feed)
#             layer_output = layer_output + r.tolist()
#         X_adv_acts.append(layer_output)
#
#     lid_dim = len(model.layers) + 1
#     print("Number of layers to estimate: ", lid_dim)
#
#     def estimate(i_batch):
#         start = i_batch * batch_size
#         end = np.minimum(len(X), (i_batch + 1) * batch_size)
#         n_feed = end - start
#
#         start_adv = i_batch * adv_batch_size
#         end_adv = np.minimum(len(X_adv), (i_batch + 1) * adv_batch_size)
#         n_feed_adv = end_adv - start_adv
#
#         lid_batch = np.zeros(shape=(n_feed, lid_dim))
#         lid_batch_adv = np.zeros(shape=(n_feed_adv, lid_dim))
#
#         for i in range(len(X_acts)):
#             X_act = np.asarray(X_acts[i][start:end], dtype=np.float32).reshape((n_feed, -1))
#             # print("X_act: ", X_act.shape)
#
#             X_adv_act = np.asarray(X_adv_acts[i][start_adv:end_adv], dtype=np.float32).reshape((n_feed_adv, -1))
#             # print("X_adv_act: ", X_adv_act.shape)
#
#             # random clean samples
#             # Maximum likelihood estimation of local intrinsic dimensionality (LID)
#             lid_batch[:, i] = mle_batch(X_act, X_act, k=k)
#             # print("lid_batch: ", lid_batch.shape)
#             lid_batch_adv[:, i] = mle_batch(X_act, X_adv_act, k=k)
#             # print("lid_batch_adv: ", lid_batch_adv.shape)
#
#         return lid_batch, lid_batch_adv
#
#     lids = []
#     lids_adv = []
#     n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
#     adv_batch_size = int(np.ceil(len(X_adv) / n_batches))
#     for i_batch in tqdm(range(n_batches)):
#         lid_batch, lid_batch_adv = estimate(i_batch)
#         lids.extend(lid_batch)
#         lids_adv.extend(lid_batch_adv)
#
#     lids = np.asarray(lids, dtype=np.float32)
#     lids_adv = np.asarray(lids_adv, dtype=np.float32)
#
#     return lids, lids_adv