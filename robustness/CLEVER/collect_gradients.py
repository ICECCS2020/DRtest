#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_gradients.py

Front end for collecting maximum gradient norm samples

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""

from __future__ import division

import numpy as np
import scipy.io as sio
import random
import time
import sys
import os
from functools import partial

from estimate_gradient_norm import EstimateLipschitz
from utils import generate_data
sys.path.append('../..')
from nmutant_model.model_operation import model_load
from nmutant_util.utils_tf import model_prediction, model_argmax
from nmutant_data.data import get_data

def collect_gradients(dataset="mnist",
                      model_name="lenet5",
                      activation="relu",
                      Nsamps=1024,
                      Niters=500,
                      numimg=1,
                      ids="",
                      target_type=0b01111,
                      firstimg=0,
                      transform="",
                      order=1,
                      compute_slope=False,
                      sample_norm="l2",
                      fix_dirty_bug=False,
                      batch_size=0,
                      nthreads=0,
                      save="./lipschitz_mat",
                      seed=1215,
                      #seed=0,
                      de=False,
                      attack='fgsm',
                      epoch=49):

    
    #args = vars(parser.parse_args())
    args={'Nsamps': Nsamps,
          'target_type': target_type,
          'batch_size': batch_size,
          'save': save,
          'numimg': numimg,
          'seed': seed,
          'order': order,
          'dataset': dataset,
          'firstimg': firstimg,
          'transform': transform,
          'Niters': Niters,
          'nthreads': nthreads,
          'sample_norm': sample_norm,
          'activation': activation,
          'model_name': model_name,
          'ids': ids,
          'fix_dirty_bug': fix_dirty_bug,
          'compute_slope': compute_slope}
    f=open('arg.txt', 'w')
    f.write(str(args))
    
    seed = args['seed']
    Nsamp = args['Nsamps'];
    Niters = args['Niters'];
    dataset = args['dataset']
    model_name = args['model_name']
    start = args['firstimg']
    numimg = args['numimg']
    save_path = args['save']
    total = 0

    random.seed(seed)
    np.random.seed(seed)

    # create output directory
    os.system("mkdir -p {}/{}_{}".format(save_path, dataset, model_name))

    # create a Lipschitz estimator class (initial it early to save multiprocessing memory)
    clever_estimator = EstimateLipschitz(sess=None, nthreads=args['nthreads'])

    # import the ID lists
    if args['ids']:
        import pandas as pd
        df = pd.read_csv(args['ids'], sep = "\t")
        # don't use this
        if args['fix_dirty_bug']:
            df = df[df['new_class'] != df['target']]
        ids = list(df['id'])
        target_type = args['target_type']
        # use the target classes override
        if target_type & 0b1000 != 0:
            target_classes = [[t] for t in list(df['target'])]
        else:
            # use generated classes
            target_classes = None
    else:
        ids = None
        target_classes = None
        target_type = args['target_type']

    import tensorflow as tf
    from setup_cifar import CIFAR
    from setup_mnist import MNIST
    from setup_imagenet import ImageNet
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        clever_estimator.sess = sess
        # returns the input tensor and output prediction vector
        img, output = clever_estimator.load_model(dataset, model_name, activation = args['activation'], batch_size = args['batch_size'], compute_slope = args['compute_slope'], order = args['order'], de=de, attack=attack, epoch=epoch)
        # load dataset

        datasets_loader = {"mnist": MNIST, "cifar": CIFAR, "imagenet": partial(ImageNet, clever_estimator.model.image_size)}
        data = datasets_loader[dataset]()


        predictor = lambda x: np.squeeze(sess.run(output, feed_dict = {img: x}))
        # generate target images
        inputs, targets, true_labels, true_ids, img_info = generate_data(data, samples=numimg, targeted=True,
                                    start=start, predictor=predictor,
                                    random_and_least_likely = True,
                                    ids = ids, target_classes = target_classes, target_type = target_type,
                                    imagenet="imagenet" in dataset,
                                    remove_background_class="imagenet" in dataset and 
                                    ("vgg" in model_name or "densenet" in model_name or "alexnet" in model_name))

        timestart = time.time()
        print("got {} images".format(inputs.shape))
        for i, input_img in enumerate(inputs):
            # original_predict = np.squeeze(sess.run(output, feed_dict = {img: [input_img]}))
            print("processing image {}".format(i))
            original_predict = predictor([input_img])
            true_label = np.argmax(true_labels[i])
            predicted_label = np.argmax(original_predict)
            least_likely_label = np.argmin(original_predict)
            original_prob = np.sort(original_predict)
            original_class = np.argsort(original_predict)
            print("Top-10 classifications:", original_class[-1:-11:-1])
            print("True label:", true_label)
            print("Top-10 probabilities/logits:", original_prob[-1:-11:-1])
            print("Most unlikely classifications:", original_class[:10])
            print("Most unlikely probabilities/logits:", original_prob[:10])
            if true_label != predicted_label:
                print("[WARNING] This image is classfied wrongly by the classifier! Skipping!")
                continue
            total += 1
            # set target class
            target_label = np.argmax(targets[i]);
            print('Target class: ', target_label)
            sys.stdout.flush()
           
            if args['order'] == 1:
                [L2_max,L1_max,Li_max,G2_max,G1_max,Gi_max,g_x0,pred] = clever_estimator.estimate(input_img, true_label, target_label, Nsamp, Niters, args['sample_norm'], args['transform'], args['order'])
                print("[STATS][L1] total = {}, seq = {}, id = {}, time = {:.3f}, true_class = {}, target_class = {}, info = {}".format(total, i, true_ids[i], time.time() - timestart, true_label, target_label, img_info[i]))
                # save to sampling results to matlab ;)
                mat_path = "{}/{}_{}/{}_{}_{}_{}_{}_{}_{}_order{}".format(save_path, dataset, model_name, Nsamp, Niters, true_ids[i], true_label, target_label, img_info[i], args['activation'], args['order'])
                save_dict = {'L2_max': L2_max, 'L1_max': L1_max, 'Li_max': Li_max, 'G2_max': G2_max, 'G1_max': G1_max, 'Gi_max': Gi_max, 'pred': pred, 'g_x0': g_x0, 'id': true_ids[i], 'true_label': true_label, 'target_label': target_label, 'info':img_info[i], 'args': args, 'path': mat_path}
                sio.savemat(mat_path, save_dict)
                print('saved to', mat_path)
                sys.stdout.flush()
            elif args['order'] == 2:
                [H2_max,g_x0,g_x0_grad_2_norm,g_x0_grad_1_norm,g_x0_grad_inf_norm,pred] = clever_estimator.estimate(input_img, true_label, target_label, Nsamp, Niters, args['sample_norm'], args['transform'], args['order'])
                #print("[STATS][L1] H2_max = {}, g_x0 = {:.5g}, g_x0_grad_2_norm = {:.5g}, g_x0_grad_1_norm = {:.5g}, g_x0_grad_inf_norm = {:.5g}, pred = {}".format(H2_max,g_x0,g_x0_grad_2_norm, g_x0_grad_1_norm, g_x0_grad_inf_norm, pred))
                print("[STATS][L1] total = {}, seq = {}, id = {}, time = {:.3f}, true_class = {}, target_class = {}, info = {}".format(total, i, true_ids[i], time.time() - timestart, true_label, target_label, img_info[i]))
                ### Lily TODO: save the computed quantities to mat file
                # save to sampling results to matlab ;)
                mat_path = "{}/{}_{}/{}_{}_{}_{}_{}_{}_{}_order{}".format(save_path, dataset, model_name, Nsamp, Niters, true_ids[i], true_label, target_label, img_info[i], args['activation'], args['order'])
                save_dict = {'H2_max': H2_max, 'pred': pred, 'g_x0': g_x0, 'id': true_ids[i], 'true_label': true_label, 'target_label': target_label, 'info':img_info[i], 'args': args, 'path': mat_path, 'g_x0_grad_2_norm': g_x0_grad_2_norm}
                sio.savemat(mat_path, save_dict)
                print('saved to', mat_path)
                sys.stdout.flush()
if __name__ == '__main__':
    collect_gradients(save='./lipschitz_mat/cifar_test', model_name='vgg13', dataset='cifar', numimg=5, firstimg=0, target_type=16)

