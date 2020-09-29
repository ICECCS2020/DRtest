import os
import numpy as np
import sys
sys.path.append("../")

for model in ['vgg11', 'vgg19']:
    for attack in ['fgsm', 'jsma']:#, 'cw', 'jsma'
        for mu_var in ['gf', 'nai', 'ns', 'ws']:
            os.system('CUDA_VISIBLE_DEVICES=1 python retrain_mu_mnist.py --datasets=mnist --attack=' + attack + ' --model_name=' + model + ' --mu_var=' + mu_var + ' --epochs=150')