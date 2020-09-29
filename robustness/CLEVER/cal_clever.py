#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clever.py

Compute CLEVER score using collected Lipschitz constants

Copyright (C) 2017-2018, IBM Corp.
Copyright (C) 2017, Lily Weng  <twweng@mit.edu>
                and Huan Zhang <ecezhang@ucdavis.edu>

This program is licenced under the Apache 2.0 licence,
contained in the LICENCE file in this directory.
"""

from clever import clever_score

dataset='mnist'
models=['vgg13']
attacks=['oritest']
istarget='target'
#clever_score(data_folder='lipschitz_mat/target/mnist_lenet1')

for model in models:
    for attack in attacks:
        clever_score(data_folder='lipschitz_mat/'+istarget+'/'+dataset+'/'+model+'/'+attack+'/'+dataset+'_'+model, untargeted=False)
